#!/usr/bin/env python
#coding: utf-8
#helper functions for the soil water balance data fusion
import os
import uuid
import warnings
from glob import glob

import numpy as np
import pandas as pd
import rioxarray
from osgeo import gdal
from scipy.stats import norm, truncnorm

warnings.filterwarnings('ignore')

#help gdal find the proj database
try:
    from pyproj import datadir as _pyproj_datadir
    os.environ['PROJ_LIB'] = _pyproj_datadir.get_data_dir()
except Exception:
    pass


#raster input/output

def map_vals_to_grid_nan(vals, valid_mask, template_shape):
    #place a vector of valid-pixel values back onto the full grid
    grid_flat = np.full(len(valid_mask), np.nan, dtype=np.float32)
    grid_flat[valid_mask] = vals
    return grid_flat.reshape(template_shape)


def save_tif(filepath, data, geo, proj):
    nrows, ncols = data.shape
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(filepath, ncols, nrows, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(geo)
    ds.SetProjection(proj)
    b = ds.GetRasterBand(1)
    b.WriteArray(data.astype(np.float32))
    b.SetNoDataValue(float('nan'))
    ds.FlushCache()
    ds = None


def load_tif_vals(filepath, valid_mask):
    with rioxarray.open_rasterio(filepath) as rds:
        return rds.squeeze().values.astype(np.float32).flatten()[valid_mask]


def load_tif_vals_match(filepath, valid_mask, template_shape, template_file):
    #load a raster, reprojecting onto the template grid if shapes differ
    with rioxarray.open_rasterio(filepath) as rds:
        if rds.squeeze().values.shape == template_shape:
            return rds.squeeze().values.astype(np.float32).flatten()[valid_mask]
        with rioxarray.open_rasterio(template_file) as tpl:
            reprojected = rds.rio.reproject_match(tpl)
    return reprojected.squeeze().values.astype(np.float32).flatten()[valid_mask]


def try_load_tif_match(filepath, valid_mask, template_shape, template_file):
    #as load_tif_vals_match but returns zeros when the file is missing
    try:
        return load_tif_vals_match(filepath, valid_mask, template_shape, template_file)
    except Exception:
        return np.zeros(int(valid_mask.sum()), dtype=np.float32)


def q_date(d):
    #yyyy.mm.dd to dd.mm.yyyy (naming convention of the runoff files)
    y, m, dd = d.split('.')
    return f'{dd}.{m}.{y}'


#input loading

def load_inputs(cfg, carry_diag_extra):
    #load every static input and assemble the initial storage state
    template_file = cfg['template_file']
    folders = cfg['folders']
    R_factors = np.load(cfg['lowrank_npz'])
    U_R, s_R, valid_mask = (R_factors['U'].astype(np.float64),
                            R_factors['s'].astype(np.float64),
                            R_factors['valid_mask'])
    n_valid = int(valid_mask.sum())
    phi = np.ones(n_valid, dtype=np.float64) / n_valid

    tds = gdal.Open(template_file)
    ts = tds.ReadAsArray().astype(np.float32).shape
    geo, proj = tds.GetGeoTransform(), tds.GetProjection()

    scalars = pd.read_csv(cfg['scalars_csv']).set_index('date')

    df_smap = pd.read_excel(cfg['srz_obs_xlsx'])
    df_smap['date'] = pd.to_datetime(df_smap['date']).dt.strftime('%Y.%m.%d')
    mObsSsw_series = dict(zip(df_smap['date'], df_smap['mSrzObs']))
    vObsSsw_series = dict(zip(df_smap['date'], df_smap['sSrzObs'].pow(2)))

    #initial storage S0: ESA CCI storage data product one month before the start of the analysis period
    #mean (bias) and random error per grid cell, covariance written as an outer product of vector standard deviations s_S0 s_S0'
    mPriorSsw0 = np.nan_to_num(load_tif_vals_match(
        cfg['s0_mean_tif'], valid_mask, ts, template_file), nan=0.0)
    sPriorSsw0 = np.nan_to_num(np.abs(load_tif_vals_match(
        cfg['s0_std_tif'], valid_mask, ts, template_file)), nan=0.0).astype(np.float64)

    #S0 is part of the joint prior x_t = {S0, P, C, E, Qrain, Qirr, F}, p(x_t) ~ N(m_x, V_x)
    #prior covariance in low-rank form V_S0 = W diag(Lam) W' = s_S0 s_S0' (W = s_S0 as one column, Lam = 1), plus the diagonal D_S0 below
    W_Ssw0 = sPriorSsw0[:, None]
    Lam_Ssw0 = np.ones(1, dtype=np.float64)
    #additional independent (sub-grid) variance D_S0 = b s_S0^2, b = 1 in run_datafusion.py
    D_Ssw0 = (carry_diag_extra * sPriorSsw0 ** 2).astype(np.float64)


    #residual and saturated soil moisture theta_res, theta_sat; S_min = theta_res Z, S_max = theta_sat Z; -9999 nodata replaced by the median
    def load_theta(path):
        v = np.nan_to_num(load_tif_vals_match(path, valid_mask, ts, template_file),
                          nan=0.0).astype(np.float64)
        bad = (v < -1.0) | (v > 2.0)
        return np.where(bad, np.median(v[~bad]), v)

    theta_res_f = load_theta(cfg['theta_res_tif'])
    theta_sat_f = load_theta(cfg['theta_sat_tif'])
    Rz = 1000.0
    Smin_soil = theta_res_f * Rz
    Smax_soil = theta_sat_f * Rz

    common_dates = sorted([
        os.path.basename(f)[-14:-4]
        for f in glob(os.path.join(folders['emax'], '*.tif'))
    ])
    common_dates = [d for d in common_dates if d in mObsSsw_series]
    max_dates = os.environ.get('max_dates')
    if max_dates:
        common_dates = common_dates[:int(max_dates)]
    print(f'{len(common_dates)} dates  ({common_dates[0]} to {common_dates[-1]})\n')
    return (U_R, s_R, valid_mask, n_valid, phi, ts, geo, proj, scalars,
            mObsSsw_series, vObsSsw_series, mPriorSsw0, W_Ssw0, Lam_Ssw0,
            D_Ssw0, Smin_soil, Smax_soil, common_dates)


#low-rank gaussian algebra

def extract_lowrank(W, M):
    #orthonormal factors (U, s) with W M W' = U diag(s) U'
    #eigendecomposition of M directly, in case M is psd
    eigvals, eigvecs = np.linalg.eigh(M)
    tol_e = 1e-12 * max(float(np.abs(eigvals).max()), 1.0)
    pos = eigvals > tol_e
    A = W @ eigvecs[:, pos]
    B = A * np.sqrt(eigvals[pos])

    BtB = 0.5 * (B.T @ B + (B.T @ B).T)
    eta, V = np.linalg.eigh(BtB)
    idx = np.argsort(eta)[::-1]
    eta, V = eta[idx], V[:, idx]

    tol2 = 1e-12 * max(float(eta[0]) if len(eta) > 0 else 1.0, 1.0)
    pos2 = eta > tol2
    s_out = eta[pos2]
    U_out = B @ V[:, pos2] / np.sqrt(s_out)

    return U_out.astype(np.float32), s_out.astype(np.float32)


def extract_lowrank_v2(W, M):
    #orthonormal factors (U, s) with W M W' = U diag(s) U'
    #QR of W first, then eigendecomposition, in case M has a negative sign
    #(e.g. the stacked storage prior: E and the runoffs enter M with minus signs)
    Q, R = np.linalg.qr(W)
    Mp = R @ M @ R.T
    Mp = 0.5 * (Mp + Mp.T)
    eigvals, eigvecs = np.linalg.eigh(Mp)
    tol = 1e-9 * max(float(np.abs(eigvals).max()), 1.0)
    pos = eigvals > tol
    U_out = Q @ eigvecs[:, pos]
    s_out = eigvals[pos]
    return U_out.astype(np.float64), s_out.astype(np.float64)


def avg_V(phi, U=None, s=None, D=None, W=None, M=None):
    #basin mean variance phi' V phi for a covariance held in factored form
    v = 0.0
    if D is not None:
        v += float(np.sum(phi ** 2 * D))
    if U is not None and s is not None and U.size:
        c = U.T.astype(np.float64) @ phi
        v += float(np.sum(np.asarray(s, dtype=np.float64) * c ** 2))
    if W is not None and M is not None and W.size:
        c = W.T.astype(np.float64) @ phi
        v += float(c @ np.asarray(M, dtype=np.float64) @ c)
    return max(v, 0.0)


def anomaly_lowrank(m, D, U, Lam, q):
    #Gaussian multiplication: combining independent information on the soil water storage anomaly with the grid-scale filtered posterior of S
    n = m.shape[0]
    qi = 1.0 / q
    Di = 1.0 / D
    DiU = Di[:, None] * U
    C = np.linalg.inv(np.diag(1.0 / Lam) + U.T @ DiU)

    r = U.shape[1]
    W_p = np.hstack([DiU, np.ones((n, 1)), qi[:, None]])
    M_p = np.zeros((r + 2, r + 2))
    M_p[:r, :r] = -C
    M_p[r, r] = qi.sum() / n ** 2
    M_p[r, r + 1] = M_p[r + 1, r] = -1.0 / n

    p0 = Di + qi
    ip0 = 1.0 / p0
    W_v = ip0[:, None] * W_p
    #M-free Woodbury (a singular M_p is harmless). note the ordering:
    #(I + M B)^-1 M, not M (I + M B)^-1, bcz M and B do not commute
    B = W_p.T @ W_v
    M_v = -np.linalg.solve(np.eye(r + 2) + M_p @ B, M_p)
    M_v = 0.5 * (M_v + M_v.T)

    #mean: m_post = m - V_post A diag(1/q) A m
    t = m - m.mean()
    t = t * qi
    t = t - t.mean()
    m_post = m - (ip0 * t + W_v @ (M_v @ (W_v.T @ t)))
    return m_post, ip0, W_v, M_v


#stacked covariance factors

def build_stack(blocks, lams, kP, kE, w_C, v_PE, v_PC, v_EC):
    #factors (W, M) of the joint covariance over the given variable blocks
    #plus the P-E-C cross-coupling columns; the cross block is signed
    W = np.hstack(blocks + [kP[:, None], kE[:, None], w_C[:, None]])
    sizes = [blk.shape[1] for blk in blocks] + [1, 1, 1]
    off = np.cumsum([0] + sizes)
    M = np.zeros((off[-1], off[-1]))
    for i, lam in enumerate(lams):
        M[off[i]:off[i + 1], off[i]:off[i + 1]] = np.diag(lam)
    i_kP, i_kE, i_wC = off[len(blocks)], off[len(blocks) + 1], off[len(blocks) + 2]
    M[i_kP, i_kE] = M[i_kE, i_kP] = -v_PE   #sign_P*sign_E
    M[i_kP, i_wC] = M[i_wC, i_kP] = +v_PC   #sign_P*sign_C
    M[i_kE, i_wC] = M[i_wC, i_kE] = -v_EC   #sign_E*sign_C
    return W, M


#expectation propagation and box truncation

_infernet_ready = False


def _setup_infernet():
    global _infernet_ready
    if _infernet_ready:
        return
    os.environ['PYTHONNET_RUNTIME'] = 'coreclr'
    from pythonnet import load as pyload
    pyload('coreclr')
    import clr
    nuget = os.path.expanduser('~/.nuget/packages')

    for pattern, fname in [
        ('microsoft.codeanalysis.csharp', 'Microsoft.CodeAnalysis.CSharp.dll'),
        ('microsoft.codeanalysis.common', 'Microsoft.CodeAnalysis.dll'),
    ]:
        hits = glob(os.path.join(nuget, pattern, '**', fname), recursive=True)
        for dll in hits:
            if any(tag in dll for tag in ('netstandard2.0', 'net6.0', 'net8.0')):
                clr.AddReference(dll)
                break

    for subpath in [
        'microsoft.ml.probabilistic/0.4.2504.701/lib/netstandard2.0/Microsoft.ML.Probabilistic.dll',
        'microsoft.ml.probabilistic.compiler/0.4.2504.701/lib/netstandard2.0/Microsoft.ML.Probabilistic.Compiler.dll',
    ]:
        clr.AddReference(os.path.join(nuget, subpath))

    _infernet_ready = True


def constrain_between(mu_array, var_array, lower, upper):
    #per-pixel truncated-normal moment matching via Infer.NET
    _setup_infernet()
    from Microsoft.ML.Probabilistic.Models import Variable, Range, InferenceEngine
    from System import Array, Double

    n = len(mu_array)
    uid = uuid.uuid4().hex[:8]

    data_range = Range(n).Named(f'n_{uid}')
    means = Variable.Observed[Double](
        Array[Double](mu_array.astype(float)), data_range).Named(f'means_{uid}')
    variances = Variable.Observed[Double](
        Array[Double](np.maximum(var_array, 1e-10).astype(float)), data_range).Named(f'variances_{uid}')
    #lower/upper may be a scalar (flat box) or a per-pixel array ; broadcast_to covers both
    lowers = Variable.Observed[Double](
        Array[Double](np.broadcast_to(np.asarray(lower, dtype=float), (n,)).copy()),
        data_range).Named(f'lowers_{uid}')
    uppers = Variable.Observed[Double](
        Array[Double](np.broadcast_to(np.asarray(upper, dtype=float), (n,)).copy()),
        data_range).Named(f'uppers_{uid}')

    x = Variable.Array[Double](data_range).Named(f'x_{uid}')
    x[data_range] = Variable.GaussianFromMeanAndVariance(means[data_range], variances[data_range])
    Variable.ConstrainBetween(x[data_range], lowers[data_range], uppers[data_range])

    engine = InferenceEngine()
    engine.ShowProgress = False
    engine.ModelName = f'trunc_{uid}'

    post_array = engine.Infer(x)
    mu_post = np.array([post_array[i].GetMean() for i in range(n)], dtype=np.float32)
    var_post = np.array([post_array[i].GetVariance() for i in range(n)], dtype=np.float32)
    return mu_post, var_post


def _init_tau_eta_from_1d(mu, v, lo, hi):
    #warm-start site parameters from the per-pixel truncated-normal moments
    sigma = np.sqrt(np.maximum(v, 1e-10))
    #lo/hi scalar or per-pixel
    alpha = (lo - mu) / sigma
    beta = (hi - mu) / sigma
    pdf_a = norm.pdf(alpha)
    pdf_b = norm.pdf(beta)
    Z = np.maximum(norm.cdf(beta) - norm.cdf(alpha), 1e-15)
    m_tr = mu + sigma * (pdf_a - pdf_b) / Z
    v_tr = np.maximum(v * (1.0 + (alpha * pdf_a - beta * pdf_b) / Z
                           - ((pdf_a - pdf_b) / Z) ** 2), 1e-10)
    return 1.0 / v_tr - 1.0 / v, m_tr / v_tr - mu / v


def apply_ep_truncation(mu, U, s, lo, hi, n_iterations=50, tol=1e-3):
    #expectation propagation truncation of N(mu, U diag(s) U') to [lo, hi]
    #computed using the expectation propagation (EP) inference technique (Minka, 2012)
    mu0 = mu.astype(np.float64)
    U0 = U.astype(np.float64)
    s0 = s.astype(np.float64)
    s0_inv = 1.0 / np.maximum(s0, 1e-30)
    diag_V0 = np.sum((U0 ** 2) * s0, axis=1)

    def compute_posterior(tau, eta):
        M = 0.5 * ((np.diag(s0_inv) + (U0.T * tau) @ U0) +
                   (np.diag(s0_inv) + (U0.T * tau) @ U0).T)
        eigv, eigvec = np.linalg.eigh(M)
        inv_eig = 1.0 / np.maximum(eigv, max(float(s0_inv.min()) * 1e-8, 1e-15))

        def Minv(B):
            return eigvec @ ((eigvec.T @ B) * inv_eig[:, None] if B.ndim == 2
                             else (eigvec.T @ B) * inv_eig)

        A_y = eta - tau * (U0 @ Minv(U0.T @ eta))
        A_mu = tau * mu0 - tau * (U0 @ Minv(U0.T @ (tau * mu0)))
        mu_p = mu0 + U0 @ (s0 * (U0.T @ (A_y - A_mu)))

        T = 0.5 * ((U0.T * tau) @ U0 + ((U0.T * tau) @ U0).T)
        Z = 0.5 * (T - T @ Minv(T) + (T - T @ Minv(T)).T)
        SZS = s0[:, None] * Z * s0[None, :]
        v_p = np.maximum(diag_V0 - np.sum((U0 @ SZS) * U0, axis=1), 1e-12)
        return mu_p, v_p, Z

    #initialise the site parameters from the exact per-pixel marginal
    #truncated-normal moments, then iterate damped EP
    tau_lik, eta_lik = _init_tau_eta_from_1d(mu0, diag_V0, lo, hi)
    mu_post, v_post, Z_cur = compute_posterior(tau_lik, eta_lik)

    n_iters_done = 0
    delta_rel = np.nan
    converged = False
    damp = 0.5   #removing the damping makes strongly constrained months diverge

    for t in range(n_iterations):
        n_iters_done = t + 1

        #posterior natural parameters
        prec_post = 1.0 / np.maximum(v_post, 1e-12)
        nat_post = mu_post * prec_post

        #cavity: subtract the current site from the posterior
        prec_cav_raw = prec_post - tau_lik
        valid = prec_cav_raw > 1e-8
        prec_cav = np.where(valid, prec_cav_raw, prec_post)
        nat_cav = np.where(valid, nat_post - eta_lik, nat_post)

        #moment-matched truncation of the cavity marginals
        m_trunc, v_trunc = constrain_between(
            (nat_cav / prec_cav).astype(np.float32),
            (1.0 / prec_cav).astype(np.float32), lo, hi)
        m_trunc = m_trunc.astype(np.float64)
        v_trunc = np.maximum(v_trunc.astype(np.float64), 1e-10)
        prec_trunc = 1.0 / v_trunc

        #damped site update
        tau_new = damp * (prec_trunc - prec_cav) + (1.0 - damp) * tau_lik
        eta_new = damp * (m_trunc * prec_trunc - nat_cav) + (1.0 - damp) * eta_lik
        tau_lik = np.where(valid, tau_new, (1.0 - damp) * tau_lik)
        eta_lik = np.where(valid, eta_new, (1.0 - damp) * eta_lik)

        #recompute the posterior from the fixed prior and updated sites
        mu_old = mu_post
        mu_post, v_post, Z_cur = compute_posterior(tau_lik, eta_lik)
        delta_rel = np.linalg.norm(mu_post - mu_old) / (np.linalg.norm(mu_old) + 1e-15)
        if delta_rel < tol and n_iters_done >= 4:
            converged = True
            break

    #extract posterior low-rank factors from the final site parameters
    P = 0.5 * ((np.diag(s0) - s0[:, None] * Z_cur * s0[None, :]) +
               (np.diag(s0) - s0[:, None] * Z_cur * s0[None, :]).T)
    eigvals_P, eigvecs_P = np.linalg.eigh(P)
    idx = np.argsort(eigvals_P)[::-1]
    eigvals_P = eigvals_P[idx]
    eigvecs_P = eigvecs_P[:, idx]
    pos = eigvals_P > 1e-15

    return (mu_post.astype(np.float32),
            (U0 @ eigvecs_P[:, pos]).astype(np.float32),
            eigvals_P[pos].astype(np.float32),
            n_iters_done, converged, delta_rel)


#box truncation

def truncate_box(m, v, lo, hi):
    #per-pixel truncated-normal moment matching of N(m, v) to the box [lo, hi]
    sd = np.sqrt(np.maximum(v, 1e-12))
    a = (lo - m) / sd
    b = (hi - m) / sd
    m_tr = truncnorm.mean(a, b, loc=m, scale=sd)
    v_tr = truncnorm.var(a, b, loc=m, scale=sd)
    return m_tr.astype(np.float64), np.maximum(v_tr, 1e-12).astype(np.float64)

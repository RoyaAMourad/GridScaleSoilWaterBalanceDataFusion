#!/usr/bin/env python
#coding: utf-8
#inference for the monthly grid scale soil water balance fusion
#run_datafusion.py builds the config and calls run_fusion

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from tqdm import tqdm

from error_models import load_flux_priors, prior_factors
from helpers import (map_vals_to_grid_nan, save_tif, extract_lowrank,
                     extract_lowrank_v2, avg_V, anomaly_lowrank,
                     apply_ep_truncation, truncate_box, build_stack,
                     load_inputs)


#thresholds and tunables
flux_box = (0.0, 1e12)    #nonnegativity box for P and E
ep_std_threshold = 0.5    #EP trigger std (mm/month)
ep_p_thresh_e = 0.10      #EP trigger probability for E
box_thresh = 0.10         #S0 box trigger probability
anomaly_cv = 0.2          #cv of the storage anomaly prior
carry_diag_extra = 1.0    #b in the S0 diagonal D_S0 = b s_S0^2

#monthly data fusion 
def run_fusion(cfg):
    #runs the fusion at each time step; cfg: output_dir, template_file (the 250 m grid), lowrank_npz (U_R, s_R of the correlation matrix R and the basin mask),
    #s0_mean_tif, s0_std_tif (ESA CCI storage one month before the start of the analysis period), theta_res_tif, theta_sat_tif (S_min = theta_res Z, S_max = theta_sat Z),
    #scalars_csv (basin-scale variances and cross-covariance terms of P, E, C), srz_obs_xlsx (basin-scale ESA CCI soil water storage, mean and standard error),
    #folders (P, E, C, Qirr, Qrain inputs), f_prior (net groundwater flux error model)
    os.chdir(os.path.dirname(os.path.abspath(sys.argv[0])))

    template_file = cfg['template_file']
    output_dir = cfg['output_dir']
    folders = cfg['folders']
    f_prior = cfg['f_prior']
    Pmin, Pmax = flux_box
    Emin, Emax = flux_box

    (U_R, s_R, valid_mask, n_valid, phi, ts, geo, proj, scalars,
     mObsSsw_series, vObsSsw_series, mPriorSsw0, W_Ssw0, Lam_Ssw0,
     D_Ssw0, Smin_soil, Smax_soil, common_dates) = load_inputs(cfg, carry_diag_extra)

    os.makedirs(output_dir, exist_ok=True)

    series = []
    processed = errors = 0

    for i, date_str in enumerate(tqdm(common_dates, desc='joint data fusion')):
        print(f'\n-- {date_str}  ({i+1}/{len(common_dates)}) --')
        try:
            #load the priors of all soil water balance variables
            mPriorP, sPriorP, mPriorE, sPriorE, kP, kE, mPriorQirr, sPriorQirr, mPriorQrain, sPriorQrain, mPriorC = \
                load_flux_priors(date_str, folders, valid_mask, ts, template_file)

            mObsSsw = mObsSsw_series[date_str]
            vObsSsw = vObsSsw_series[date_str]

            #groundwater exchange flux prior: weak uninformative prior with a large spread, no independent component
            mPriorF, W_F, Lam_F = f_prior(n_valid)
            D_F = np.zeros(n_valid, dtype=np.float64)
            #V_S's diagonal: the independent component of S0
            D_tot = D_Ssw0 + D_F
            cF = W_F.T @ phi
            vF_basin = float(cF @ (Lam_F * cF))

            #basin scalars of the P, E, C joint posterior
            sc = scalars.loc[date_str]
            v_P, v_E, v_C = float(sc.vP), float(sc.vE), float(sc.vC)
            v_PE, v_PC, v_EC = float(sc.vPE), float(sc.vPC), float(sc.vEC)
            dvP = float(sc.vPriorP) - v_P
            dvE = float(sc.vPriorE) - v_E

            #rank-one canal weight, phi'w = 1
            Ibar = float(phi @ mPriorC)
            w_C = mPriorC / Ibar if Ibar > 0 else np.zeros_like(mPriorC)
            sCv = np.sqrt(max(v_C, 0.0)) * w_C

            W_P, Lam_P = prior_factors(sPriorP, float(sc.vPriorP), dvP, U_R, s_R, phi)
            W_E, Lam_E = prior_factors(sPriorE, float(sc.vPriorE), dvE, U_R, s_R, phi)

            gam_P = 1.0 - v_PE / v_P + v_PC / v_P
            gam_E = 1.0 - v_PE / v_E - v_EC / v_E
            gam_C = 1.0 + v_PC / v_C - v_EC / v_C

            #S0 prior projected to basin scale
            mPriorSsw0_basin_t = float(phi @ mPriorSsw0)
            cSsw0 = W_Ssw0.T @ phi
            vPriorSsw0_basin_t = float(cSsw0 @ (Lam_Ssw0 * cSsw0) + np.sum(phi ** 2 * D_Ssw0))

            #step 1: kalman filtering
            #1a predict and compare
            #innovation and total variance; phi'K = 1 collapses phi'V phi to scalars
            m_y_prior = mPriorP - mPriorE - mPriorQirr - mPriorQrain + mPriorC + mPriorF
            v_dS = (v_P + v_E + v_C - 2.0 * v_PE + 2.0 * v_PC - 2.0 * v_EC
                    + float(phi @ sPriorQirr) ** 2 + float(phi @ sPriorQrain) ** 2 + vF_basin)
            v_pred = vPriorSsw0_basin_t + v_dS
            v_denom = v_pred + vObsSsw
            innov = mObsSsw - mPriorSsw0_basin_t - float(phi @ m_y_prior)

            #1b update
            mPostP, U_P, lam_P, vPostP, gP = kalman_update(
                mPriorP, W_P, Lam_P, +1.0, innov, v_denom, phi,
                ep_lo=Pmin, ep_hi=Pmax, gain_scale=gam_P, tag='P')

            mPostE, U_E, lam_E, vPostE, gE = kalman_update(
                mPriorE, W_E, Lam_E, -1.0, innov, v_denom, phi,
                ep_lo=Emin, ep_hi=Emax, p_thresh=ep_p_thresh_e,
                gain_scale=gam_E, tag='E')

            #Qirr is not inferred but treated as an uncertain quantity: posterior = prior, zero gain, its variance still in v_denom
            mPostQirr = mPriorQirr
            W_Qirr = sPriorQirr[:, None].astype(np.float64)   #prior factor, still enters the S stack
            vPostQirr = (sPriorQirr ** 2).astype(np.float32)
            gQirr = np.zeros(n_valid, dtype=np.float64)

            W_Qrain = sPriorQrain[:, None].astype(np.float64)
            mPostQrain, U_Qrain, lam_Qrain, vPostQrain, gQrain = kalman_update(
                mPriorQrain, W_Qrain, np.ones(1), -1.0, innov, v_denom, phi, tag='Qrain')

            W_C = sCv[:, None].astype(np.float64)
            mPostC, U_C, lam_C, vPostC, gC = kalman_update(
                mPriorC, W_C, np.ones(1), +1.0, innov, v_denom, phi,
                gain_scale=gam_C)

            mPostF, U_F, lam_F, vPostF, gF = kalman_update(
                mPriorF, W_F, Lam_F, +1.0, innov, v_denom, phi)
            vPostF = (vPostF + D_F).astype(np.float32)

            #S0 as the 7th joint variable, same shared innovation
            mPostSsw0, U_Ssw0, lam_Ssw0, vPostSsw0, gSsw0 = kalman_update(
                mPriorSsw0, W_Ssw0, Lam_Ssw0, +1.0, innov, v_denom, phi)
            #add back the independent component (basin share ~D/n is negligible)
            vPostSsw0 = (vPostSsw0 + D_Ssw0).astype(np.float32)

            #step 2: kalman smoothing
            #2a predict
            dS_mean = (mPostP - mPostE - mPostQirr - mPostQrain
                      + mPostC + mPostF)

            phi_Vy_eff = (gP + gE + gQirr + gQrain + gC + gF).astype(np.float32)

            #S = S0 + dS from the 7 posteriors
            Ssw_mean = mPostSsw0 + dS_mean

            #joint prior covariance of S; the cross block is signed, so extract_lowrank_v2
            W_wide, M_wide = build_stack(
                [W_Ssw0, W_P, W_E, W_Qirr, W_Qrain, W_C, W_F],
                [Lam_Ssw0, Lam_P, Lam_E, np.ones(1), np.ones(1), np.ones(1), Lam_F],
                kP, kE, w_C, v_PE, v_PC, v_EC)
            U_prior, Lam_prior = extract_lowrank_v2(W_wide, M_wide)

            #Sigma_R: the same stack without the F block
            W_Rp, M_Rp = build_stack(
                [W_Ssw0, W_P, W_E, W_Qirr, W_Qrain, W_C],
                [Lam_Ssw0, Lam_P, Lam_E, np.ones(1), np.ones(1), np.ones(1)],
                kP, kE, w_C, v_PE, v_PC, v_EC)

            #shared rank-one posterior correction, same pattern as kalman_update
            phi_Vy_S_eff = (phi_Vy_eff + gSsw0).astype(np.float32)
            c_S = U_prior.T @ phi_Vy_S_eff
            M_post = np.diag(Lam_prior) - np.outer(c_S, c_S) / v_denom
            U_post, Lam_post = extract_lowrank(U_prior, M_post)

            #snapshot the prior-side diagonal before the carry overwrites it
            D_Ssw0_pri = D_Ssw0
            Ssw_mean_raw = Ssw_mean

            #2b gaussian multiplication
            #zero-mean anomaly prior A S ~ N(0, diag(s_anomaly^2)), A = I - 1 phi'
            s_anomaly = anomaly_cv * np.maximum(Ssw_mean_raw, Smin_soil)
            #V_S must be full rank: a positive independent component on every pixel
            assert bool(np.all(D_tot > 0.0)), 'V_S is not full rank this month (D_tot <= 0 somewhere)'
            #exact conditioning on the anomaly prior, in precision form
            mPostSsw, v0_anomaly, W_vb, M_vb = anomaly_lowrank(
                Ssw_mean_raw.astype(np.float64), D_tot,
                U_post, Lam_post, (s_anomaly ** 2).astype(np.float64))
            #level stays with the assimilated basin observation
            mPostSsw = mPostSsw - float(np.mean(mPostSsw - Ssw_mean_raw))
            delta_anomaly = mPostSsw - Ssw_mean_raw
            lr_diag = np.einsum('ij,jk,ik->i', W_vb, M_vb, W_vb)
            vPostSsw = np.maximum(v0_anomaly + lr_diag, 1e-6).astype(np.float32)
            #Sigma_S factors (diag(v0) + W M W') of the conditioned storage
            Sigma_S_fac = (v0_anomaly, W_vb, M_vb)

            #carry the storage as next month's rank-one prior
            mPriorSsw0 = mPostSsw
            _v_carry = np.maximum(vPostSsw, 0.0).astype(np.float64)
            W_Ssw0 = np.sqrt(_v_carry)[:, None]
            Lam_Ssw0 = np.ones(1, dtype=np.float64)
            D_Ssw0 = carry_diag_extra * _v_carry

            #2c update
            #covariance of the full stack with the basin observation, for the F closure
            gS_vec = phi_Vy_S_eff.astype(np.float64)

            variables = [('S0', mPostSsw0, vPostSsw0, +1.0),
                       ('P', mPostP, vPostP, +1.0),
                       ('E', mPostE, vPostE, -1.0),
                       ('Qirr', mPostQirr, vPostQirr, -1.0),
                       ('Qrain', mPostQrain, vPostQrain, -1.0),
                       ('C', mPostC, vPostC, +1.0),
                       ('F', mPostF, vPostF, +1.0)]

            sm_mean, sm_var, sign_of = {}, {}, {}
            #F with an uninformative prior: the Kalman gain approaches the identity matrix K -> I for F and K_k -> 0 for each other term,
            #so the smoothed posterior of each term k equals its filtered posterior and F receives the whole correction m_F = m_F-> + (m_S - m_->S)
            G_R = (gS_vec - gF).astype(np.float64)
            W_Rf = np.hstack([W_Rp, G_R[:, None]])
            M_Rf = np.zeros((M_Rp.shape[0] + 1,) * 2)
            M_Rf[:-1, :-1] = M_Rp
            M_Rf[-1, -1] = -1.0 / v_denom
            #Sigma_R: covariance of the six other terms R = S - F, before the carry; var F = diag(Sigma_R) + diag(Sigma_S)
            diag_R = D_Ssw0_pri + np.einsum('ij,jk,ik->i', W_Rf, M_Rf, W_Rf)
            diag_S = Sigma_S_fac[0] + np.einsum(
                'ij,jk,ik->i', Sigma_S_fac[1], Sigma_S_fac[2], Sigma_S_fac[1])
            varF_post = np.maximum(diag_R + diag_S, 1e-6)
            for name, m_post, v_k_own, sign in variables:
                sign_of[name] = sign
                sm_mean[name] = np.array(m_post, dtype=np.float64)
                sm_var[name] = np.asarray(v_k_own, dtype=np.float64)
            sm_mean['F'] = mPostF + delta_anomaly
            sm_var['F'] = varF_post

            s0_box_fix(sm_mean, sm_var, Smin_soil, Smax_soil, phi)

            Ssw_recombined = np.zeros(n_valid)
            for name, *_ in variables:
                Ssw_recombined = Ssw_recombined + sign_of[name] * sm_mean[name]

            #outputs and basin summary
            #posterior means and stds as GeoTIFFs
            for tag, m_arr, v_arr in ([('S', Ssw_recombined, vPostSsw)]
                    + [(nm, sm_mean[nm], sm_var[nm]) for nm, *_ in variables]):
                d = os.path.join(output_dir, f'{tag}_smoothed_tif')
                os.makedirs(d, exist_ok=True)
                save_tif(os.path.join(d, f'm{tag}smoothed_{date_str}.tif'),
                         map_vals_to_grid_nan(np.asarray(m_arr).astype(np.float32),
                                              valid_mask, ts), geo, proj)
                save_tif(os.path.join(d, f's{tag}smoothed_{date_str}.tif'),
                         map_vals_to_grid_nan(np.sqrt(np.maximum(v_arr, 0.0)).astype(np.float32),
                                              valid_mask, ts), geo, proj)

            closure = float(np.max(np.abs(Ssw_recombined - mPostSsw)))
            print(f'closure {closure:.2e} mm')
            #basin-mean std sqrt(phi' V phi) per variable
            _plus = {'S0': (U_Ssw0, lam_Ssw0), 'P': (U_P, lam_P),
                     'E': (U_E, lam_E), 'Qrain': (U_Qrain, lam_Qrain),
                     'C': (U_C, lam_C)}
            sm_basin_var = {}
            for nm, *_ in variables:
                if nm == 'F':
                    sm_basin_var[nm] = (
                        avg_V(phi, D=D_Ssw0_pri, W=W_Rf, M=M_Rf)
                        + avg_V(phi, D=Sigma_S_fac[0],
                                            W=Sigma_S_fac[1], M=Sigma_S_fac[2]))
                    continue
                if nm == 'Qirr':
                    #marginalised: rank-one prior covariance
                    sm_basin_var[nm] = float((phi @ sPriorQirr) ** 2)
                    continue
                U_k, s_k = _plus[nm]
                sm_basin_var[nm] = avg_V(
                    phi, U=U_k, s=s_k, D=(D_Ssw0_pri if nm == 'S0' else None))
            sm_basin_std_marg = {nm: float(np.sqrt(v))
                                 for nm, v in sm_basin_var.items()}

            def bm(v):
                return float(phi @ v)

            #basin posterior mean and std per variable
            row = {'date': date_str,
                   'mObsSsw': mObsSsw,
                   'mPostSsw': bm(mPostSsw),
                   'sPostSsw': float(np.sqrt(max(
                       np.sum(phi ** 2 * Sigma_S_fac[0])
                       + (Sigma_S_fac[1].T @ phi) @ (Sigma_S_fac[2]
                         @ (Sigma_S_fac[1].T @ phi)), 0.0)))}
            for nm, *_ in variables:
                col = 'Ssw0' if nm == 'S0' else nm
                row[f'mPost{col}'] = float(phi @ sm_mean[nm])
                row[f'sPost{col}'] = sm_basin_std_marg[nm]
            series.append(row)
            processed += 1

        except Exception as exc:
            import traceback
            print(f'error: {exc}')
            traceback.print_exc()
            errors += 1

    if series:
        pd.DataFrame(series).to_csv(
            os.path.join(output_dir, 'basin_summary_joint.csv'), index=False)

    print(f'\ndone  processed={processed}  errors={errors}')


#helper functions of run_fusion

#the joint update

def kalman_update(m_prior, W_k, Lam_k, sign_k, innov, v_denom, phi,
                      ep_lo=None, ep_hi=None, p_thresh=0.05,
                      gain_scale=1.0, tag=''):
    #Kalman filter update of block k of the joint prior N(m_x->, V_x->) by the likelihood N(phi' A x, v_Sobs); one call per variable
    #innovation m_Sobs - phi' A m_x->; denominator v_Sobs + phi' A V_x-> A' phi, a sum of scalar variances, so the update is rank one
    #g_k = sgn_k V_k-> phi for the independent variables; g_k = sgn_k V_k-> phi + sum_j sgn_j V_kj phi for k in {P, E, C} with cross-covariance terms (gamma_k = gain_scale)
    #m_k = m_k-> + sgn_k g_k innovation / denominator; V_k = V_k-> - g_k g_k' / denominator; sgn_k the sign of term k in the water balance
    c_k = W_k.T @ phi
    g_k = gain_scale * (W_k @ (Lam_k * c_k))

    m_post = (m_prior + sign_k * g_k * innov / v_denom).astype(np.float64)

    M_k_post = np.diag(Lam_k) - gain_scale ** 2 * \
        np.outer(Lam_k * c_k, Lam_k * c_k) / v_denom
    U_post, s_post = extract_lowrank(W_k, M_k_post)
    diag_V = np.sum((U_post ** 2) * s_post, axis=1).astype(np.float32)

    if ep_lo is not None:
        #trigger only where the std exceeds ep_std_threshold
        std_p = np.sqrt(np.maximum(diag_V, 0.0)).astype(np.float64)
        above_thresh = std_p > ep_std_threshold
        safe = np.where(above_thresh, std_p, 1.0)
        p_viol = np.where(above_thresh, norm.cdf((ep_lo - m_post) / safe), 0.0)
        trig = p_viol > p_thresh
        n_trig = int(trig.sum())
        if n_trig > 0:
            try:
                m_post, U_post, s_post, n_ep, conv, _ = apply_ep_truncation(
                    m_post, U_post, s_post, ep_lo, ep_hi)
                diag_V = np.sum((U_post ** 2) * s_post, axis=1).astype(np.float32)
                print(f'EP {tag}: {n_ep} iterations, converged={conv}, {n_trig} cells')
            except Exception as ex:
                #keep the unconstrained posterior, report the violation
                viol_p = p_viol[trig]
                worst_m = float(m_post[trig].min())
                print(f'EP {tag} failed ({type(ex).__name__}): {n_trig} cells, '
                      f'max P(violate) {viol_p.max():.3f}, min mean {worst_m:.4g}')

    return (m_post.astype(np.float32),
            U_post.astype(np.float32), s_post.astype(np.float32),
            diag_V, g_k.astype(np.float32))


#S0 physical box

def s0_box_fix(sm_mean, sm_var, Smin_soil, Smax_soil, phi):
    #S0 kept within the soil water storage bounds S_min = theta_res Z and S_max = theta_sat Z: truncated normal mean and variance for the grid cells outside the bounds
    sd0 = np.sqrt(np.maximum(sm_var['S0'], 1e-12))
    p_out = (norm.cdf((Smin_soil - sm_mean['S0']) / sd0)
             + 1.0 - norm.cdf((Smax_soil - sm_mean['S0']) / sd0))
    fix = p_out > box_thresh
    n_fix = int(fix.sum())
    if n_fix > 0:
        m_tr, v_tr = truncate_box(sm_mean['S0'][fix], sm_var['S0'][fix],
                                  Smin_soil[fix], Smax_soil[fix])
        d0 = m_tr - sm_mean['S0'][fix]
        sm_mean['S0'][fix] = m_tr
        sm_var['S0'][fix] = v_tr
        sm_mean['F'][fix] = sm_mean['F'][fix] - d0
        print(f'S0 bounds: {n_fix} cells, max shift {float(np.max(np.abs(d0))):.1f} mm')

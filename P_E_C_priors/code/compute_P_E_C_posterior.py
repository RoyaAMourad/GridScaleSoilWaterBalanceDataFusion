#!/usr/bin/env python
#coding: utf-8
#this file is used to generate the grid-scale priors of P, E and C by Kalman smoothing (section 4.2 of Mourad et al., 2026)
#Usage: python compute_P_E_C_posterior.py

import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import rioxarray
import rasterio
from rasterio.crs import CRS
from affine import Affine
from osgeo import gdal
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
try:
    from pyproj import datadir as _pyproj_datadir
    os.environ['PROJ_LIB'] = _pyproj_datadir.get_data_dir()
except Exception:
    pass

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath('../../data_fusion/code'))
from helpers import load_tif_vals, map_vals_to_grid_nan   #shared raster i/o


def save_as_tiff(path, array, geo, proj):
    with rasterio.open(path, 'w', driver='GTiff',
                       height=array.shape[0], width=array.shape[1],
                       count=1, dtype='float32', crs=CRS.from_wkt(proj),
                       transform=Affine.from_gdal(*geo), nodata=np.nan) as dst:
        dst.write(array.astype('float32'), 1)

out_dir = '../output/P_E_C_grid_priors'


#low-rank correlation structure
lr = np.load('../input/lowrank_factors.npz')
U_R, s_R   = lr['U'].astype(np.float64), lr['s'].astype(np.float64)
valid_mask = lr['valid_mask'].astype(bool)
n_valid    = int(valid_mask.sum())
print(f"low-rank R: U={U_R.shape}, n_valid={n_valid}")


#precipitation error model (Mourad et al., 2026, HESS)
#prior mean (1-wP)*Pmin + wP*Pmax; prior standard deviation rP*0.25*|Pmax-Pmin|
def precipitation_error_model(pmin_vals, pmax_vals):
    wP = 0.5051; rP = 0.9554
    mPriorP = (1 - wP) * pmin_vals + wP * pmax_vals
    sPriorP = rP * 0.25 * np.abs(pmax_vals - pmin_vals)
    return mPriorP, sPriorP


#evaporation error model (Mourad et al., 2026, HESS)
#prior mean fE*((1-wE)*Emin + wE*Emax); prior standard deviation rE*0.25*|Emax-Emin|
def evaporation_error_model(emin_vals, emax_vals):
    wE = 0.3958; fE = 0.9954; rE = 0.9593
    mPriorE = fE * ((1 - wE) * emin_vals + wE * emax_vals)
    sPriorE = rE * 0.25 * np.abs(emax_vals - emin_vals)
    return mPriorE, sPriorE


#canal irrigation error model
#grid-scale mean: the basin-scale m_C weighted by the normalized spatial weight w = w_c / mean(w_c), so phi'w = 1
#the field is perfectly correlated: covariance V_C = v_C w w', the outer product of the normalized weight scaled by the posterior basin-scale variance; sPostC = sqrt(v_C) w
def canal_irrigation_model(date_str, vBasinPostC):
    mPostC = np.nan_to_num(
        load_tif_vals(f'../input/C/canal_irr_{date_str[:7]}.tif', valid_mask), nan=0.0)
    Ibar = float(np.mean(mPostC))
    w_C = mPostC / Ibar if Ibar > 0 else np.zeros_like(mPostC)
    sPostC = (np.sqrt(vBasinPostC) * w_C).astype(np.float32)
    return mPostC, sPostC, Ibar


#Kalman smoothing from the basin-scale posteriors to the grid-scale posteriors of P and E
#Q stays basin-scale: only the 2x2 P-E block of vBasinPost enters
def compute(sPriorP, sPriorE,
            mPriorP, mPriorE,
            mBasinPrior, mBasinPost, vBasinPost_folder, date_str):

    #V_k phi via the rank-15 factors of R
    def V_phi(sPrior):
        c = U_R.T @ (sPrior / n_valid)         #r-vector
        return sPrior * (U_R @ (s_R * c))      #n-vector = V_k phi

    Vphi_P = V_phi(sPriorP.astype(np.float64))
    Vphi_E = V_phi(sPriorE.astype(np.float64))

    #basin-scale prior variances  vb_k = phi' V_k phi
    vBasinPriorP = float(np.mean(Vphi_P))
    vBasinPriorE = float(np.mean(Vphi_E))

    #vBasinPost csv: posterior covariance of the basin water-balance closure
    vBasinPost_file = os.path.join(vBasinPost_folder, f'vBasinPost_{date_str}.csv')
    vBasinPost = np.loadtxt(vBasinPost_file, delimiter=',')
    vBasinPostP = float(vBasinPost[0, 0])    #marginal basin posterior variance of P
    vBasinPostE = float(vBasinPost[1, 1])    #marginal basin posterior variance of E

    #gain  K_k = V_k phi / vb_k ;  the spatial average of K is 1
    K_matrix = np.zeros((n_valid, 2), dtype=np.float32)
    K_matrix[:, 0] = (Vphi_P / vBasinPriorP).astype(np.float32)   #K for P
    K_matrix[:, 1] = (Vphi_E / vBasinPriorE).astype(np.float32)   #K for E

    #posterior means: if we average the grid-scale posterior mean of P and E, we get back the basin-scale posterior mean
    m_innovation = mBasinPost - mBasinPrior
    mPostP = mPriorP + K_matrix[:, 0] * m_innovation[0]
    mPostE = mPriorE + K_matrix[:, 1] * m_innovation[1]

    #marginal variances; cross blocks rank one, V*_jk = v*_jk K_j K_k'
    vPostP_diag = sPriorP ** 2 + (K_matrix[:, 0] ** 2) * (vBasinPostP - vBasinPriorP)
    vPostE_diag = sPriorE ** 2 + (K_matrix[:, 1] ** 2) * (vBasinPostE - vBasinPriorE)

    #check whether the avg of SRS gives us back the basin-scale variance
    phiK_P = float(np.mean(K_matrix[:, 0]))
    phiK_E = float(np.mean(K_matrix[:, 1]))
    avgCovPostP = vBasinPriorP + phiK_P ** 2 * (vBasinPostP - vBasinPriorP)
    avgCovPostE = vBasinPriorE + phiK_E ** 2 * (vBasinPostE - vBasinPriorE)
    print(f"{date_str}: avg variance - basin variance  P {avgCovPostP - vBasinPostP:+.3e}  E {avgCovPostE - vBasinPostE:+.3e}")

    return (mPostP, vPostP_diag, mPostE, vPostE_diag,
            K_matrix, vBasinPost, vBasinPriorP, vBasinPriorE)


#per-month driver and scalar table

template_file = '../input/Emax/Emax_2003.01.01.tif'
with rioxarray.open_rasterio(template_file) as rds:
    template_data = rds.squeeze().values.astype(np.float32)
dest_array = gdal.Open(template_file)
geo  = dest_array.GetGeoTransform()
proj = dest_array.GetProjection()

emax_folder = '../input/Emax'
emin_folder = '../input/Emin'
pmax_folder = '../input/Pmax'
pmin_folder = '../input/Pmin'

emax_files   = sorted(glob(os.path.join(emax_folder, '*.tif')))
common_dates = [os.path.basename(f)[-14:-4] for f in emax_files]

#results<X>.out: posteriors of the basin water-balance closure, parsed to mPost<X>/vPost<X>
data_path = os.path.abspath('../input/basin_posteriors')
posteriors_files = [(f, open(f).read()) for f in glob(os.path.join(data_path, '*.out'))
                    if os.path.basename(f) not in ('resultsPar.out', 'resultsS0.out')]
posteriors = {}
for file, data in posteriors_files:
    key_suffix = os.path.splitext(os.path.basename(file))[0].split('results', 1)[-1]
    parsed = [tuple(map(float, e.strip('Gaussian()').replace('(', '').replace(')', '').split(',')))
              for e in data.split('\n') if e and 'Gaussian' in e]
    if parsed:
        posteriors['mPost' + key_suffix] = [v[0] for v in parsed]
        posteriors['vPost' + key_suffix] = [v[1] for v in parsed]
dfPosteriors = pd.DataFrame(posteriors)

#export the basin-scale posteriors for inspection
dfP_out = dfPosteriors.copy()
if len(dfP_out) == len(common_dates):
    dfP_out.insert(0, 'date', common_dates)
os.makedirs('../output', exist_ok=True)
dfP_out.to_excel('../output/dfPosteriors.xlsx', index=False)
print(f"wrote ../output/dfPosteriors.xlsx  shape={dfP_out.shape}")


def process_single_date(date_str):
    date_idx = common_dates.index(date_str)

    def load(folder, prefix):
        return load_tif_vals(os.path.join(folder, f'{prefix}_{date_str}.tif'), valid_mask)

    emax_vals = load(emax_folder, 'Emax');  emin_vals = load(emin_folder, 'Emin')
    pmax_vals = load(pmax_folder, 'Pmax');  pmin_vals = load(pmin_folder, 'Pmin')

    mPriorP, sPriorP = precipitation_error_model(pmin_vals, pmax_vals)
    mPriorE, sPriorE = evaporation_error_model(emin_vals, emax_vals)

    mBasinPrior = np.array([np.mean(mPriorP), np.mean(mPriorE)], dtype=np.float32)
    mBasinPost  = np.array([dfPosteriors.loc[date_idx, 'mPostP'],
                            dfPosteriors.loc[date_idx, 'mPostE']], dtype=np.float32)

    (mPostP, vPostP_diag, mPostE, vPostE_diag,
     K_matrix, vBasinPost, vBasinPriorP, vBasinPriorE) = compute(
        sPriorP=sPriorP, sPriorE=sPriorE,
        mPriorP=mPriorP, mPriorE=mPriorE,
        mBasinPrior=mBasinPrior, mBasinPost=mBasinPost,
        vBasinPost_folder='../input/cross_covariances',
        date_str=date_str)

    sPostP = np.sqrt(np.maximum(vPostP_diag, 0.0)).astype(np.float32)
    sPostE = np.sqrt(np.maximum(vPostE_diag, 0.0)).astype(np.float32)

    vBasinPostC = float(vBasinPost[3, 3])          #order [P,E,Q,C]
    mPostC, sPostC, Ibar = canal_irrigation_model(date_str, vBasinPostC)

    #sPrior, the R factors, K and the scalars give V* = V + K (v* - v) K', V = S R S
    for nm, vv in [('mPostP', mPostP), ('sPostP', sPostP),
                   ('mPostE', mPostE), ('sPostE', sPostE),
                   ('mPostC', mPostC), ('sPostC', sPostC),
                   ('sPriorP', sPriorP),    ('sPriorE', sPriorE),
                   ('KP', K_matrix[:, 0]),  ('KE', K_matrix[:, 1])]:
        os.makedirs(os.path.join(out_dir, nm), exist_ok=True)
        save_as_tiff(os.path.join(out_dir, nm, f'{nm}_{date_str}.tif'),
                     map_vals_to_grid_nan(vv, valid_mask, template_data.shape), geo, proj)

    print(f"[{date_str}] mPostP={mPostP.mean():.1f} sPostP=[{sPostP.min():.1f},"
          f"{sPostP.max():.1f}]  mPostE={mPostE.mean():.1f} "
          f"sPostE=[{sPostE.min():.1f},{sPostE.max():.1f}]  "
          f"mPostC={Ibar:.1f} sPostC_basin={np.sqrt(vBasinPostC):.2f} "
          f"(cv={np.sqrt(vBasinPostC)/max(Ibar,1e-9):.3f})")

    #one scalar row per month; with KP/KE/mPostC these give V*_jk = v*_jk K_j K_k'
    scalar_rows.append({
        'date':     date_str,
        'vP':       float(vBasinPost[0, 0]),
        'vE':       float(vBasinPost[1, 1]),
        'vC':       float(vBasinPost[3, 3]),
        'vPE':      float(vBasinPost[0, 1]),
        'vPC':      float(vBasinPost[0, 3]),
        'vEC':      float(vBasinPost[1, 3]),
        'vPriorP':  vBasinPriorP,
        'vPriorE':  vBasinPriorE,
        'Ibar':     Ibar,
    })


scalar_rows = []

if __name__ == "__main__":
    n_max = os.environ.get('max_dates')
    dates_to_run = common_dates[:int(n_max)] if n_max else common_dates
    print(f"running {len(dates_to_run)} of {len(common_dates)} dates to {out_dir}")
    for date_str in tqdm(dates_to_run, desc="P/E posterior"):
        process_single_date(date_str)
    if scalar_rows:
        pd.DataFrame(scalar_rows).to_csv(
            os.path.join(out_dir, 'basin_joint_scalars.csv'), index=False)
        print(f"wrote {out_dir}/basin_joint_scalars.csv ({len(scalar_rows)} rows)")
    print("done")

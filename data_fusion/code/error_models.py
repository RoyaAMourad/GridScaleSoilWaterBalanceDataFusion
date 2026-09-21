#!/usr/bin/env python
#coding: utf-8
#functions to generate the grid-scale error models for each term in the soil water balance

import os
import numpy as np
from helpers import load_tif_vals, load_tif_vals_match, try_load_tif_match, q_date

#prior means and standard deviations for all soil water balance variables

def load_flux_priors(date_str, folders, valid_mask, ts, template_file):
    #grid-scale priors: P, E, C from compute_P_E_C_posterior.py; Qirr, Qrain from run_discharge_closure.py
    #error model: m_x = m_x_obs, V_x = S_x R S_x, R low rank U_R diag(s_R) U_R', r = 15
    #kalman smoothing: m* = m + K (m*bar - mbar), V* = V + K (v*bar - vbar) K', K = V phi / vbar
    qd = q_date(date_str)
    lm = lambda p: load_tif_vals_match(p, valid_mask, ts, template_file)
    tm = lambda p: try_load_tif_match(p, valid_mask, ts, template_file)

    mPriorP = np.nan_to_num(lm(os.path.join(folders['mpostp'], f'mPostP_{date_str}.tif')), nan=0.0).astype(np.float64)
    sPriorP = np.nan_to_num(np.abs(lm(os.path.join(folders['spriorp'], f'sPriorP_{date_str}.tif'))), nan=0.0).astype(np.float64)
    mPriorE = np.nan_to_num(lm(os.path.join(folders['mposte'], f'mPostE_{date_str}.tif')), nan=0.0).astype(np.float64)
    sPriorE = np.nan_to_num(np.abs(lm(os.path.join(folders['spriore'], f'sPriorE_{date_str}.tif'))), nan=0.0).astype(np.float64)
    kP = np.nan_to_num(lm(os.path.join(folders['kp'], f'KP_{date_str}.tif')), nan=0.0).astype(np.float64)
    kE = np.nan_to_num(lm(os.path.join(folders['ke'], f'KE_{date_str}.tif')), nan=0.0).astype(np.float64)
    #Qirr >= 0
    mPriorQirr = np.maximum(np.nan_to_num(lm(os.path.join(folders['qirr'], f'mPostQirr_{qd}.tif')), nan=0.0), 0.0).astype(np.float64)
    sPriorQirr = np.nan_to_num(np.abs(tm(os.path.join(folders['sqirr'], f'sPostQirr_{qd}.tif'))), nan=0.0).astype(np.float64)
    mPriorQrain = np.nan_to_num(lm(os.path.join(folders['qrain'], f'mPostQrain_{qd}.tif')), nan=0.0).astype(np.float64)
    sPriorQrain = np.nan_to_num(np.abs(tm(os.path.join(folders['sqrain'], f'sPostQrain_{qd}.tif'))), nan=0.0).astype(np.float64)
    mPriorC = np.nan_to_num(load_tif_vals(
        os.path.join(folders['c'], f'canal_irr_{date_str[:7]}.tif'), valid_mask), nan=0.0).astype(np.float64)
    return mPriorP, sPriorP, mPriorE, sPriorE, kP, kE, mPriorQirr, sPriorQirr, mPriorQrain, sPriorQrain, mPriorC


#prior covariance factors for P and E

def prior_factors(sDiag, vPrior, dv, U_R, s_R, phi):
    #low-rank form of the corrected grid-scale covariance: V* = W diag(lam) W'
    #V = S R S, R = U_R diag(s_R) U_R'; V* = V + K (v*bar - vbar) K', K = V phi / vbar
    #vbar = phi' V phi, basin-scale variance; phi = 1/n, spatial average; psd if v*bar >= 0
    W0 = sDiag[:, None] * U_R
    if vPrior <= 0.0:
        return W0, s_R.copy()
    a = s_R * (W0.T @ phi) / vPrior
    M0 = np.diag(s_R) - dv * np.outer(a, a)
    lam, Q = np.linalg.eigh(M0)
    keep = lam > 1e-14 * max(float(lam.max()), 1.0)
    return (W0 @ Q[:, keep]).astype(np.float64), lam[keep].astype(np.float64)


#groundwater exchange flux error model

sPriorF = 500.0   #prior standard deviation of F (mm) with a wide spread
def f_prior(n_valid):
    #mean 0 and random error 500 at every grid, spatially uniform: V_F = s_F s_F'
    mPriorF = np.zeros(n_valid, dtype=np.float64)
    W_F = np.full((n_valid, 1), sPriorF, dtype=np.float64)
    Lam_F = np.ones(1, dtype=np.float64)
    return mPriorF, W_F, Lam_F

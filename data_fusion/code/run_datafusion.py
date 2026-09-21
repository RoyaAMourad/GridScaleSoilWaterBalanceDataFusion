#!/usr/bin/env python
#coding: utf-8
#this file is used to run the grid scale soil water balance data fusion
#Usage: python run_datafusion.py

import error_models
from solver import run_fusion

config = {
    'output_dir':    '../output',
    'theta_res_tif': '../input/theta_res.tif',
    'theta_sat_tif': '../input/theta_sat.tif',
    'template_file': '../input/WaPOR_ETa_2003.01.01.tif',
    'lowrank_npz':   '../input/lowrank_factors.npz',
    's0_mean_tif':   '../input/ESA_CCI_Srz_EndDayofMonth_2002.12.01.tif',
    's0_std_tif':    '../input/ESA_CCI_Srz_Uncertainty_EndDayofMonth_2002.12.01.tif',
    'scalars_csv':   '../../P_E_C_priors/output/P_E_C_grid_priors/basin_joint_scalars.csv',
    'srz_obs_xlsx':  '../input/SrzObs.xlsx',
    'folders': {
        'mpostp':  '../../P_E_C_priors/output/P_E_C_grid_priors/mPostP',
        'spriorp': '../../P_E_C_priors/output/P_E_C_grid_priors/sPriorP',
        'mposte':  '../../P_E_C_priors/output/P_E_C_grid_priors/mPostE',
        'spriore': '../../P_E_C_priors/output/P_E_C_grid_priors/sPriorE',
        'kp':      '../../P_E_C_priors/output/P_E_C_grid_priors/KP',
        'ke':      '../../P_E_C_priors/output/P_E_C_grid_priors/KE',
        'qirr':    '../../Qprior/output/mPostQirr',
        'sqirr':   '../../Qprior/output/sPostQirr',
        'qrain':   '../../Qprior/output/mPostQrain',
        'sqrain':  '../../Qprior/output/sPostQrain',
        'c':       '../input/C',
        'emax':    '../input/Emax',
    },
}
config['f_prior'] = error_models.f_prior

if __name__ == '__main__':
    run_fusion(config)

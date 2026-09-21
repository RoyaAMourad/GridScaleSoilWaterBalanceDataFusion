#this file is used to decompose the basin discharge posterior into its components and to downscale Qrain and Qirr to the grid
#Usage: python run_discharge_closure.py
#we use the calibrated river discharge Q_obs output of a previous posterior analysis (Mourad et al., 2026) as a likelihood
#for the sum of all prior components; the Qrain and Qirr posteriors written here are the Q priors of the grid fusion
import re
import subprocess
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path

#input loading

here = Path(__file__).parent          #Qprior/code
inp  = here.parent / 'input'
outd = here.parent / 'output'
outd.mkdir(exist_ok=True)

#resultsQ.out: the calibrated river discharge of Mourad et al. (2026)
gauss_re = re.compile(r'Gaussian\(\s*([\d.eE+\-]+),\s*([\d.eE+\-]+)\)')
post_means, post_vars = [], []
with open(inp / 'resultsQ.out') as fh:
    for line in fh:
        m = gauss_re.search(line)
        if m:
            post_means.append(float(m.group(1)))
            post_vars.append(float(m.group(2)))

gcn_means = pd.read_csv(inp / 'GCN250_basin_mean.csv',
                        parse_dates=['date'], float_precision='round_trip')
dates = [d.strftime('%Y.%m.%d') for d in gcn_means['date']]
n = min(len(post_means), len(dates))
df_post = pd.DataFrame({
    'date': pd.to_datetime([d.replace('.', '-') for d in dates[:n]]),
    'Qposterior': post_means[:n],
    'vQposterior': post_vars[:n],
}).set_index('date')

#component input series: domestic, industrial, escape
df_dom = pd.read_csv(inp / 'Qdomestic.csv', parse_dates=['date']).set_index('date')
df_ind = pd.read_csv(inp / 'Qindustry.csv', parse_dates=['date']).set_index('date')
df_esc = (pd.read_excel(inp / 'Qescape.xlsx', parse_dates=['date'])
            .set_index('date')[['Qescape (mm/month)']]
            .rename(columns={'Qescape (mm/month)': 'Qescape'}))
df = df_post.join(df_dom).join(df_ind).join(df_esc)
df['Qescape'] = df['Qescape'].fillna(0.0)

gcn_dir = inp / 'GCN250_Galeta_masked'
rain_basin_prior = dict(zip(gcn_means['date'], gcn_means['Qrain_prior']))
df['Qrain_prior'] = df.index.map(rain_basin_prior)

#error models: prior mean and standard deviation per discharge component
#the mu_*/sig_* (and mean_*/var_*) basin-scalar names mirror the priors/posteriors csv columns read by the C# closure model

#Qrain, Qesc, Qbf are treated as unknown random variables to be updated by the discharge closure constraint,
#while Qirr, Qdom, Qind are not inferred but treated as uncertain quantities (marginalized out, posterior = prior)
#standard deviations: 0.5 times the mean for Qrain; 0.3 times the mean for Qdom and Qind; 0.05 times Qesc for years with
#flow measurements (2021, 2022) and 0.5 otherwise; Qbf mean 0 with 5 mm/month post-2014 and 50 mm/month pre-2014
sig_bf_on  = 50.0   #mm, pre-2014
sig_bf_off =  5.0   #mm, post-2014
sig_rain = 0.50

#Qirr is negligible for the studied basin due to farmers' management practices involving boundary bund structures around
#agricultural fields: a time-invariant variable with mean zero and random error 5 mm/month
mu_irr  = 0.0    #mm/month
sig_irr = 5.0    #mm
sig_dom = 0.30   #cv on the Qdomestic series
sig_ind = 0.30   #cv on the Qindustry series

measured_esc_years = {2021, 2022}
sig_esc_vals = np.where(
    df.index.year.isin(measured_esc_years),
    0.05 * np.maximum(df['Qescape'], 1e-3),
    0.50 * np.maximum(df['Qescape'], 1e-3),
)

pre2014 = df.index.year < 2014
sig_bf = np.where(pre2014, sig_bf_on, sig_bf_off)

priors = pd.DataFrame({
    'date':     df.index.strftime('%Y-%m-%d'),
    'mu_rain':  df['Qrain_prior'],
    'sig_rain': np.maximum(sig_rain * df['Qrain_prior'], 0.01),
    'mu_irr':   mu_irr,
    'sig_irr':  sig_irr,
    'mu_dom':   df['Qdomestic'],
    'sig_dom':  sig_dom * df['Qdomestic'],
    'mu_ind':   df['Qindustry'],
    'sig_ind':  sig_ind * df['Qindustry'],
    'mu_esc':   df['Qescape'],
    'sig_esc':  sig_esc_vals,
    'y_obs':     df['Qposterior'],    #soft closure constraint: ConstrainEqualRandom(sumQ, N(y_obs,v_obs)) in C#
    'v_obs':     df['vQposterior'],   #upstream Q posterior variance
    'mu_bf':  0.0,                 #col 13
    'sig_bf': sig_bf,          #col 14 (read positionally by the C# closure model)
})
priors_path = outd / 'priors.csv'
priors.to_csv(priors_path, index=False)
print(f'wrote {len(priors)} rows to {priors_path.name}')

#discharge closure: marginal Gaussian distributions for each of the inferred components in each month,
#computed using the expectation propagation (EP) inference technique (Minka, 2012), Program.cs
posts_path = outd / 'posteriors.csv'
print('running the dotnet discharge closure ...')
closure_args = ['dotnet', 'run', '--project', str(here), '--', str(priors_path), str(posts_path),
                '--fixed-irr-systematic', '--fixed-dom', '--fixed-ind']
subprocess.run(closure_args, check=True)

#closure check: sum of posterior component means vs the basin posterior
df_p = pd.read_csv(posts_path, parse_dates=['date']).set_index('date')
df_p['sum_post'] = df_p[['mean_rain','mean_irr','mean_dom','mean_ind','mean_esc','mean_bf']].sum(axis=1)
df_p['Qposterior'] = df['Qposterior'].values
df_p['closure_res'] = df_p['sum_post'] - df_p['Qposterior']
print(f'closure max abs residual:  {df_p["closure_res"].abs().max():.6f} mm/month')
print(f'closure mean abs residual: {df_p["closure_res"].abs().mean():.6f} mm/month')

df_p.to_csv(outd / 'closure_check.csv')

#downscaling: kalman smoothing relations relate the basin-scale to the grid-scale quantities (Mourad et al., 2026)
#   m* = m + K (m*bar - mbar), V* = s s' + K (v*bar - vbar) K', K = s / sbar
#Qrain: the GCN250 runoff is the grid-scale prior mean, the standard deviation 0.5 times the mean
#Qirr: specified at 0 at each grid cell; the spatial information enters via its random error, the normalized GLC100
#cropland fraction fc/mean(fc) scaling the basin-scale random error sig_irr

cv_rain = 0.50
crop_dir = inp / 'cropland_fraction'

dirs = {
    'mPostQrain': outd / 'mPostQrain',
    'sPostQrain': outd / 'sPostQrain',
    'mPostQirr':  outd / 'mPostQirr',
    'sPostQirr':  outd / 'sPostQirr',
}
for d in dirs.values():
    d.mkdir(exist_ok=True)

def kalman_update(m_prior, s_diag, mu_post, v_post, valid):
    s_bar = float(np.nanmean(s_diag[valid]))
    v_basin_prior = s_bar ** 2
    if s_bar == 0 or v_basin_prior == 0:
        m_out = np.where(valid, mu_post, np.nan).astype(np.float32)
        s_out = np.where(valid, np.sqrt(v_post), np.nan).astype(np.float32)
        return m_out, s_out
    K = np.where(valid, s_diag / s_bar, 0.0)
    innov = mu_post - float(np.nanmean(m_prior[valid]))
    m_out = np.where(valid, m_prior + K * innov, np.nan)
    v_out = np.where(valid, s_diag ** 2 * v_post / v_basin_prior, np.nan)
    s_out = np.sqrt(np.where(valid, v_out, np.nan))
    return m_out.astype(np.float32), s_out.astype(np.float32)

for date, row in df_p.iterrows():
    yyyy, mm, dd = date.year, f'{date.month:02d}', f'{date.day:02d}'
    tag = f'{dd}.{mm}.{yyyy}'

    #skip months without a GCN250 raster
    if not (gcn_dir / f'Q_{tag}.tif').exists():
        continue

    #Qrain on the GCN grid
    with rasterio.open(gcn_dir / f'Q_{tag}.tif') as src:
        mPriorQrain = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nd = src.nodata
        if nd is not None:
            mPriorQrain[mPriorQrain == nd] = np.nan
        dst_transform, dst_crs = src.transform, src.crs
        H, W = src.height, src.width
    profile.update(dtype='float32', nodata=np.nan, count=1)
    valid_r = ~np.isnan(mPriorQrain)

    sPriorQrain = cv_rain * np.where(valid_r, mPriorQrain, 0.0)
    mPostQrain, sPostQrain = kalman_update(mPriorQrain, sPriorQrain, row['mean_rain'], row['var_rain'], valid_r)

    #GLC100 cropland fraction reprojected onto the GCN grid; GLC100 is available from 2015 to 2019,
    #the other years are held at the nearest available year
    crop_year = min(max(date.year, 2015), 2019)
    with rasterio.open(crop_dir / f'crops_frac_{crop_year}.tif') as src_c:
        src_c_arr = src_c.read(1).astype(np.float32)
        nd_c = src_c.nodata
        if nd_c is not None:
            src_c_arr[src_c_arr == nd_c] = np.nan
        src_c_crs = src_c.crs if src_c.crs else dst_crs
        src_c_transform = src_c.transform
    fv = np.full((H, W), np.nan, dtype=np.float32)
    reproject(source=src_c_arr, destination=fv,
              src_transform=src_c_transform, src_crs=src_c_crs,
              dst_transform=dst_transform, dst_crs=dst_crs,
              resampling=Resampling.bilinear,
              src_nodata=np.nan, dst_nodata=np.nan)
    fv[~valid_r] = np.nan
    valid_i = ~np.isnan(fv) & (fv > 0)
    fv_mean = float(np.nanmean(fv[valid_i])) if valid_i.any() else 0.0

    if fv_mean > 0:
        #zero-mean prior, std carries the cropland pattern: s(x) = sig_irr * fC/avg(fC)
        pattern = np.where(valid_i, fv / fv_mean, 0.0).astype(np.float32)
        mPriorQirr = np.zeros_like(pattern)
        sPriorQirr = sig_irr * pattern
        mPostQirr, sPostQirr = kalman_update(mPriorQirr, sPriorQirr, row['mean_irr'], row['var_irr'], valid_i)
    else:
        mPostQirr = np.full_like(mPriorQrain, np.nan)
        sPostQirr = np.full_like(mPriorQrain, np.nan)

    out_files = {
        dirs['mPostQrain'] / f'mPostQrain_{tag}.tif': mPostQrain,
        dirs['sPostQrain'] / f'sPostQrain_{tag}.tif': sPostQrain,
        dirs['mPostQirr']  / f'mPostQirr_{tag}.tif':  mPostQirr,
        dirs['sPostQirr']  / f'sPostQirr_{tag}.tif':  sPostQirr,
    }
    for path, arr in out_files.items():
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(arr, 1)

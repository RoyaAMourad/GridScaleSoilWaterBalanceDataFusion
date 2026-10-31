# Grid-scale soil water balance data fusion
[![DOI](https://zenodo.org/badge/1438360750.svg)](https://doi.org/10.5281/zenodo.23835749)

This repository provides a tool for grid-scale soil water balance data fusion of a river basin used for :
* estimating all spatial variables in the monthly 250m grid-scale 1-meter depth soil water balance model;
* adjusting the original (prior) data sets describing each water balance variable to obtain their error-corrected (calibrated) versions.

## Tools

* `lowrank_factors` folder contains the following tool:
  * `compute_lowrank_factors.py`: computes the low rank factors of the correlation matrix using the randomized SVD algorithm ([Brunton and Kutz, 2022](#brunton2022); [Halko et al., 2011](#halko2011)).
* `P_E_C_priors` folder contains the following tool:
  * `compute_P_E_C_posterior.py`: performs the grid-scale Kalman smoothing update of $P$ and $E$ described in section 4.2 of [Mourad et al. (2026)](#mourad2026).
* `Qprior` folder contains the following tools:
  * `Program.cs`: contains the river discharge balance model in Infer.NET.
  * `run_discharge_closure.py`:
    * defines the prior means and standard deviations of all discharge components;
    * runs `Program.cs`;
    * uses a Kalman smoothing technique that relates the calibrated basin-scale estimated rainfall and irrigation runoff to their grid-scale quantities to obtain their grid-scale priors.
* `data_fusion` folder contains the grid-scale soil water balance data fusion model:
  * `error_models.py`: defines the grid-scale error models.
  * `solver.py`: computes the posteriors of all variables using the Kalman filter and smoothing algorithms.
  * `helpers.py`: contains functions for low-rank Gaussian algebra, the Infer.NET truncation, and other helper functions.
  * `run_datafusion.py`: runs the grid-scale soil water balance data fusion.


## Input data

The inputs of precipitation, evaporation, and canal irrigation data sets are calibrated estimates obtained from a previous analysis that applies a total water balance data fusion methodology to combine and jointly calibrate the multi-scale water balance variables ([Mourad et al., 2026](#mourad2026)).

The posteriors of the basin-scale total water balance data fusion are input files here: `P_E_C_priors/input/basin_posteriors/`, `P_E_C_priors/input/cross_covariances/`, `Qprior/input/resultsQ.out`.

The river discharge file `resultsQ.out` is not shared here because the discharge data are restricted. To run `run_discharge_closure.py`, put your own monthly river discharge in `Qprior/input/resultsQ.out`, one month per line, written as `Gaussian(mean, variance)`.

Run the tools by typing in the terminal/console (.NET 8, Python 3.11.5 with the packages in `requirements.txt`):

```bash
python Qprior/code/run_discharge_closure.py
cd P_E_C_priors/code && python compute_P_E_C_posterior.py && cd ../..
cd data_fusion/code && python run_datafusion.py
```
The code relies on the Infer.NET probabilistic programming library (https://github.com/dotnet/infer), which is downloaded automatically (as nuget package) when the program is run for the first time. The .NET packages that `data_fusion` needs are listed in `requirements.txt`.

## Naming

`mPriorX`, `sPriorX` prior mean and standard deviation of $X$; `mPostX`, `sPostX` posterior mean and standard deviation; `vX` variance; `mObsX` observation.

## References

<a id="brunton2022"></a>Brunton, S. L., and Kutz, J. N. (2022). Data-driven science and engineering: Machine learning, dynamical systems, and control. Cambridge University Press. doi: [10.1017/9781009089517](https://doi.org/10.1017/9781009089517)

<a id="halko2011"></a>Halko, N., Martinsson, P.-G., and Tropp, J. A. (2011). Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM Review 53, 217–288. doi: [10.1137/090771806](https://doi.org/10.1137/090771806)

<a id="mourad2026"></a>Mourad, R., Schoups, G., Rajendran, V., and Bastiaanssen, W.: Joint calibration of multi-scale hydrological data sets using probabilistic water balance data fusion: methodology and application to the irrigated Hindon River Basin, India, Hydrol. Earth Syst. Sci., 30, 525–551, https://doi.org/10.5194/hess-30-525-2026, 2026

<a id="schoups2021"></a>Schoups, G., and Nasseri, M. (2021). GRACEfully closing the water balance: a data-driven probabilistic approach applied to river basins in Iran. Water Resources Research, 57, e2020WR029071, https://doi.org/10.1029/2020WR029071.

# Cross correlation Gaussian lines search algorithm for X-ray spectrums
This repository contains an implementation in python 3 of a Gaussian line search algorithm for X-ray spectra using the cross-correlation method developed by Kosec et al. 2021 in this paper : https://arxiv.org/pdf/2109.14683.pdf

### What this repository contains

The user can find 3 distinct groups of files in this repository:

- `/cross_correlation_search.py/` and `/cross_correlation_search_script.py/`. In these files you can find the implementation of the cross correlation method as well as an example of a script performing a simple search on a spectrum.

- `/compare_methods.py/` and `/compare_methods_script.py/`. These files allow the user to compare the cross correlation method to the direct fitting method by plotting the re-normalized cross-correlation (RC) against the square root of the delta Cstat. 

- `/classic_search/` is a modified version of the direct fitting method that searches for both emission and absorption lines at the same time and plot everything together. This file is required to use the `/compare_methods.py/` and `/compare_methods_script.py/` files.

An empty `/output-files/` directory is provided as well as an example. This is the folder the cross correlation algorithm will generate to store the results.

### Extensions required

- `/numpy/` and `/pandas/` for handling files containing data
- `/os/` and `/shutil/` for handling files and paths
- `/time/` for performance measurement
- `/matplotlib/` for plots using the SciencePlots module. Just type `> pip install SciencePlots`. (details at https://github.com/garrettj403/SciencePlots)
- `/PyXspec/` for spectral fitting
- `/scipy.special/` for the calculation of the significance (error function)

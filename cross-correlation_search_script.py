from xspec import *
from cross_correlation_search import *
import time

# xcm file containing processed data and an appropriate continuum model
xcm_file = 'b_004_010_groupconstant_3.xcm'

path_to_burst_spectra = '/Users/lucas/Documents/IRAP/CrossCorrelationSearch/data/BURST_SPECTRA/burst_0173sig_num1_1050300108_intervals_for1000counts/grouping_step1_bin7'

# Parameters of the search
line_widths = np.array(
    [0.02, 0.03, 0.04, 0.05, 0.06, 0.065, 0.07, 0.075, 0.08])
dE = 0.01
E_min = 0.4
E_max = 4.0
line_energies = np.arange(E_min, E_max, dE)
nb_simulations = 10000

start_time = time.time()

cross_correlation_search(path_to_burst_spectra, xcm_file,
                         False, nb_simulations, True, E_min, E_max, dE, *line_widths)

finish_time = time.time()

plot_results(path_to_burst_spectra, xcm_file,
             nb_simulations, E_min, E_max, dE, *line_widths)

print("--- %s seconds ---" % (finish_time - start_time))

from numpy import sqrt
from cross_correlation_search import *
from classic_search import *
import shutil

compare_dir = '/Users/lucas/Documents/IRAP/CrossCorrelationSearch/compare_methods'


def compare_methods(path_to_results: str, path_to_burst_spectra: str, xcm_file: str, nb_spectra: int, addLine: bool, param: list, plot: bool, nb_simulations: int, E_min: float, E_max: float, dE: float, *line_widths):
    '''
    This function verifies that the cross correlation methods yields the same results as the classic search method. More 
    precisely, the function plots RC for each gaussian parameter bin against sqrt(delta_C-stat). The result should be close 
    to y=x.

    Args:
        path_to_results (str):
            string representing the path to where you want the csv files to be saved
        path_to_bust_spectra (str):
            string representing the path to the burst spectra
        xcm_file (str) :
            xcm file containing the data and the model on which the controlled sample will be based
        nb_spectra (int):
            number of fake spectra to generate in the controlled sample
        addLine (bool):
            do you want to add a fake line in the controlled sample ?
        param (list):
            parameters for the fake line [lineE,sigma,norm]
        plot (bool):
            should the function plot the graph ?
        nb simulations (int): 
            number of simulations to perform 
        E_min (float): 
            Minimum energy to search for lines, in keV
        E_max (float): 
            Maximum energy to search for lines, in keV
        dE (float): 
            Energy step to search for lines, in keV
        line_widths (float):
            Line widths to be searched for
    '''

    # Create nb xcm files containing the fake data and the model. Controlled sample will be generated based on xcm_file
    Xset.allowPrompting = False
    for k in range(nb_spectra):
        os.chdir(path_to_burst_spectra)

        # Restore the file of reference
        Xset.restore(xcm_file)
        Fit.perform()

        # get noticed channels in the reference spectrum
        noticed_ch = AllData(1).noticedString()

        if addLine:
            # create FakeitSettings object so that the file names are consistent
            name = 'CompareMethods-line' + \
                str(param[1]*100) + 'eV-' + 'FakeSpectum-' + str(k) + '.fak'
            fs = FakeitSettings(fileName=name)

            # Add a line to the model
            AllModels(1).setPars({21: param[0]}, {22: param[1]})
            AllModels(1).gaussian.norm.values = (
                param[2], 0.001, -100, -10, 10, 100)
            AllModels(1).gaussian.norm.frozen = True

            # Generate fake data
            AllData.fakeit(1, [fs])

            # make sure that the same channels are noticed in real and fake spectra
            AllData.ignore('1-**')
            AllData.notice(noticed_ch)

            # Now change the model back by removing the gaussian line (gives the illusion of a line in the data)
            AllModels(1).gaussian.norm.values = (
                0.0, 0.001, -100, -10, 10, 100)
            AllModels(1).gaussian.norm.frozen = True

            # Save the spectrum as a xcm file
            Xset.save('CompareMethods-withLine-FakeSpectrum-' + str(k) + '.xcm')

        else:

            # create FakeitSettings object so that the file names are consistent
            name = 'CompareMethods-FakeSpectum-' + str(k) + '.fak'
            fs = FakeitSettings(fileName=name)
            AllData.fakeit(1, [fs])

            # make sure that the same channels are noticed in real and fake spectra
            AllData.ignore('1-**')
            AllData.notice(noticed_ch)

            # Save the spectrum as a xcm file
            Xset.save('CompareMethods-FakeSpectrum-' + str(k) + '.xcm')

    for k in range(nb_spectra):
        if addLine:
            file_name = 'CompareMethods-withLine-FakeSpectrum-' + \
                str(k) + '.xcm'
        else:
            file_name = 'CompareMethods-FakeSpectrum-' + str(k) + '.xcm'

        # Perform the cross correlation search on the file of the controlled sample
        cross_correlation_search(path_to_burst_spectra, file_name,
                                 nb_simulations, True, E_min, E_max, dE, *line_widths)

        # Save of copy of the file containing the normalized cc in the compare_methods_dir
        original = os.path.join(cc_dir, 'normalized-cc-real-sigma-0.csv')
        target = os.path.join(
            path_to_results, 'ccSearch-fakeSpectrum-' + str(k) + '.csv')

        shutil.copyfile(original, target)

        # Perform the cross classic search on the file of the controlled sample
        db_scan_residuals_for_lines(
            path_to_burst_spectra, file_name, True, line_widths[0], E_min, E_max, dE, False)

        # Move the result file to the compare_methods_dir and rename it
        original = os.path.join(classic_dir, file_name.replace(
            '.xcm', '') + '_LineSearch.csv')
        target = os.path.join(
            path_to_results, 'classicSearch-fakeSpectrum-' + str(k) + '.csv')

        shutil.move(original, target)

    if plot:
        compare_methods_plot(path_to_results, nb_spectra)


def compare_methods_plot(path_to_results: str, nb_spectra: int):

    os.chdir(path_to_results)
    plt.style.use(['science', 'no-latex'])

    fig, ax = plt.subplots()

    for k in range(nb_spectra):

        # Read and load the file containing the values of sqrt(delta_Cstat)
        classic_search_results = pd.read_csv(os.path.join(
            path_to_results, 'classicSearch-fakeSpectrum-' + str(k) + '.csv'), dtype=np.float64).to_numpy()
        sqrt_delta_cstat = classic_search_results[:, 6]

        # Read and load the file containing the values of the normalized cc
        cc_search_results = pd.read_csv(os.path.join(
            path_to_results, 'ccSearch-fakeSpectrum-' + str(k) + '.csv'), dtype=np.float64).to_numpy()
        normalized_cc = cc_search_results.flatten()

        # Plot everything on the graph
        ax.scatter(sqrt_delta_cstat, normalized_cc, s=7, color='red',
                   marker='D', edgecolors='black', linewidths=0.5)

    # Plot y=x
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against each other
    ax.plot(lims, lims, 'k-', zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Set the labels
    ax.set_ylabel('Renormalized cross-correlation', size=7)
    ax.set_xlabel('Direct fitting method - $\sqrt{\Delta Cstat}$', size=7)

    fig.savefig(os.path.join(path_to_results, 'CompareMethods-sampleSize-' +
                str(nb_spectra) + '.eps'), format='eps')


def compare_methods_plot_binned(path_to_results: str, nb_spectra: int):

    os.chdir(path_to_results)
    plt.style.use(['science', 'no-latex'])

    fig, ax = plt.subplots()

    # Create a data frame to hold the coordinates of each point
    df = pd.DataFrame(columns=['sqrt_delta_cstat', 'renormalized_cc'])

    for k in range(nb_spectra):

        # Read and load the file containing the values of sqrt(delta_Cstat)
        classic_search_results = pd.read_csv(os.path.join(
            path_to_results, 'classicSearch-fakeSpectrum-' + str(k) + '.csv'), dtype=np.float64).to_numpy()
        sqrt_delta_cstat = classic_search_results[:, 6]

        # Read and load the file containing the values of the normalized cc
        cc_search_results = pd.read_csv(os.path.join(
            path_to_results, 'ccSearch-fakeSpectrum-' + str(k) + '.csv'), dtype=np.float64).to_numpy()
        normalized_cc = cc_search_results.flatten()

        df_row = pd.DataFrame({'sqrt_delta_cstat': sqrt_delta_cstat,
                              'renormalized_cc': normalized_cc}, dtype=np.float64)
        df = pd.concat([df, df_row], ignore_index=True, axis=0)

    # Now sort the data frame by ascending values of srqt(delta_C-stat)
    df = df.sort_values(by='sqrt_delta_cstat')
    df = df.reset_index()

    print(df)

    # data frame to hold all the binned values
    df_binned = pd.DataFrame(
        columns=['sqrt_delta_cstat', 'renormalized_cc_mean', 'renormalized_cc_std'])
    step = 0.25

    bin_df = df[df['sqrt_delta_cstat'] == 0]
    df_row = pd.DataFrame({'sqrt_delta_cstat': [bin_df['sqrt_delta_cstat'].max()], 'renormalized_cc_mean': [
                          bin_df['renormalized_cc'].mean()], 'renormalized_cc_std': [bin_df['renormalized_cc'].std()]}, dtype=np.float64)
    df_binned = pd.concat([df_binned, df_row], ignore_index=True, axis=0)

    df_pos = df[df['sqrt_delta_cstat'] > 0]

    # bounds of the bins
    inf_bound = df_pos['sqrt_delta_cstat'].min()
    sup_bound = inf_bound + step

    while sup_bound < df_pos['sqrt_delta_cstat'].max():
        bin_df = df[(inf_bound < df['sqrt_delta_cstat']) &
                    (df['sqrt_delta_cstat'] < sup_bound)]
        if len(bin_df) < 25:
            sup_bound += step
        else:
            df_row = pd.DataFrame({'sqrt_delta_cstat': [bin_df['sqrt_delta_cstat'].max()], 'renormalized_cc_mean': [
                                  bin_df['renormalized_cc'].mean()], 'renormalized_cc_std': [bin_df['renormalized_cc'].std()]}, dtype=np.float64)
            df_binned = pd.concat([df_binned, df_row],
                                  ignore_index=True, axis=0)
            inf_bound = sup_bound
            sup_bound += step

    df_neg = df[df['sqrt_delta_cstat'] < 0]

    # bounds of the bins
    inf_bound = df_neg['sqrt_delta_cstat'].max()
    sup_bound = inf_bound - step

    while sup_bound > df_neg['sqrt_delta_cstat'].min():
        bin_df = df[(inf_bound > df['sqrt_delta_cstat']) &
                    (df['sqrt_delta_cstat'] > sup_bound)]
        if len(bin_df) < 25:
            sup_bound -= step
        else:
            df_row = pd.DataFrame({'sqrt_delta_cstat': [bin_df['sqrt_delta_cstat'].min()], 'renormalized_cc_mean': [
                                  bin_df['renormalized_cc'].mean()], 'renormalized_cc_std': [bin_df['renormalized_cc'].std()]}, dtype=np.float64)
            df_binned = pd.concat([df_binned, df_row],
                                  ignore_index=True, axis=0)
            inf_bound = sup_bound
            sup_bound -= step

    sqrt_delta_cstat = df_binned['sqrt_delta_cstat'].to_numpy()
    renormalized_cc = df_binned['renormalized_cc_mean'].to_numpy()
    y_errors = df_binned['renormalized_cc_std'].to_numpy()

    ax.errorbar(sqrt_delta_cstat, renormalized_cc, yerr=y_errors,
                linestyle='-', marker='D', ms=3, mfc='r', mec='black', mew=0.5)

    # Plot y=x
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]

    # now plot both limits against each other
    ax.plot(lims, lims, 'k-', zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Set the labels
    ax.set_ylabel('Renormalized cross-correlation', size=7)
    ax.set_xlabel('Direct fitting method - $\sqrt{\Delta Cstat}$', size=7)

    fig.savefig(os.path.join(path_to_results, 'CompareMethods-sampleSize-' +
                str(nb_spectra) + 'binned' + '.eps'), format='eps')

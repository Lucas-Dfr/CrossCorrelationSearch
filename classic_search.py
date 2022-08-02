#import db_spectral_analysis_library as dsal
from optparse import OptionParser
import os
import time
from unittest import result
import astropy.io.fits as pyfits
from optparse import OptionParser
from datetime import datetime
from astropy.io import fits
import numpy as np
import pandas as pd
import xspec
from xspec import AllData,AllModels,Xset,Spectrum,Fit,Plot,Model
import glob
#import db_tools as dt
#import db_datagraph_tools as ddt
#import db_yml_tools as dyt
from distutils.util import strtobool
#import db_manipulate_xspec_models as dmxm
import optparse
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from pathlib import Path

# Users should change this lines so that they matches the correct directory
output_dir = Path(
    '/Users/lucas/Documents/IRAP/CrossCorrelationSearch/output-files')

# No need to change this
classic_dir = os.path.join(output_dir, 'classic-search')
    
def db_scan_residuals_for_lines(path_to_burst_spectra,xcmfile, scan_line=True, line_width=0.05, emin_scan_line=0.3,emax_scan_line=3.0, step_scan_line=0.1,figure = True):
    os.chdir(path_to_burst_spectra)

    Fit.query="no"
    if scan_line :
        # LineE of the gaussian line varied from minEnergy to maxEnergy with step dE
        minEnergy=emin_scan_line
        maxEnergy=emax_scan_line
        dE=step_scan_line
        lineE_list = np.arange(minEnergy,maxEnergy,dE)

        # Norm of the gaussian line varied from 0 to 1.0 with step dNorm
        minNorm = -1.
        maxNorm=1.0
        n=40
        dNorm=(maxNorm-minNorm)/np.float(n+1)
        Norm_tested=np.linspace(maxNorm,minNorm,n+2)
        
        # Get the cstat of the best fit
        Xset.restore(xcmfile)
        Fit.perform()
        cstat_bf=Fit.statistic
        AllModels.show()
        
        # List of size len(lineE_list) to contain all the best fit cstat values
        cstat = []
        
        #input("Enter")
        Plot.device = "/xs"
        str_tp=""
        
        # Numpy matrix containing the results of each gaussian line fit
        all_results = np.array([[]])
        
        for energy in lineE_list:
            # result is an array define as np.array([sigma, lineE, norm, cstat_bf, new_cstat, delta_cstat, srqt(delta_cstat)])
            results = np.zeros(8)
            results[0] = line_width
            results[1] = energy
            results[3] = cstat_bf
            
            Xset.restore(xcmfile)
            Fit.perform()
            
            # Set the parameter of the gaussian line: only the norm will vary during the fit
            AllModels(1).gaussian.Sigma.frozen=True
            AllModels(1).gaussian.LineE.frozen=True
            AllModels(1).gaussian.norm.frozen=False
            AllModels(1).gaussian.LineE.values = (energy,-1)
            AllModels(1).gaussian.Sigma.values = (line_width)
            AllModels(1).gaussian.norm.values = (0,0.001,-100,-10,10,100)
            AllModels.show()
            
            # Perform the fit and append the results to the list of cstat
            Fit.steppar(f"{AllModels(1).gaussian.norm.index} {maxNorm} {minNorm} {n}")
            cstat.append(Fit.stepparResults("statistic"))
            
            cstat_array_int=Fit.stepparResults("statistic")
            print(cstat_array_int,"cstat-16",cstat_bf-16)
            
            # Value of the norm which minimizes the cstat
            results[2] = Norm_tested[np.argmin(cstat_array_int)]
            
            # Value of the minimum cstat reached
            results[4] = np.amin(cstat_array_int)
            
            # Delta cstat and significance
            results[5] = results[3]-results[4]
            results[6] = np.sign(results[2])*np.sqrt(results[5]) if results[5] > 0 else 0
            
            # equivalent width
            AllModels(1).setPars({12:results[2]})
            AllModels.eqwidth(4)
            results[7] = AllData(1).eqwidth[0]
            
            # Stack all the results to prepare the csv file
            all_results = np.vstack((all_results, results)) if all_results.size else results
        
            
            if np.min(cstat_array_int) < cstat_bf - 9. :
                Plot.xAxis = "KeV"
                Plot("data residuals")

                print("Minimum reached at Energy ",energy," norm = ",Norm_tested[np.argmin(cstat_array_int)], " Delta-Cstat=",np.min(cstat_array_int) - cstat_bf)
                AllModels(1).gaussian.LineE.frozen=True
                AllModels(1).gaussian.Sigma.frozen=True
                AllModels(1).gaussian.norm.frozen=False
                AllModels(1).gaussian.norm.values = (Norm_tested[np.argmin(cstat_array_int)])
                Fit.perform()
                
                print("Leaving all parameters free : ", AllModels(1).gaussian.LineE.values[0], AllModels(1).gaussian.Sigma.values[0], AllModels(1).gaussian.norm.values[0], "cstat=",Fit.statistic,"Previous statistic=",cstat_bf)
                str_tp+=",".join(str(tp) for tp in [AllModels(1).gaussian.LineE.values[0], AllModels(1).gaussian.Sigma.values[0], AllModels(1).gaussian.norm.values[0], Norm_tested[np.argmin(cstat_array_int)], Fit.statistic, np.min(cstat_array_int),Fit.statistic-cstat_bf,cstat_bf])+"\n"
#                input("Enter")
                Plot("data residuals")
#                input("Enter")

        # Turn all_results into a pd DataFrame and save it a csv file
        df_all_results = pd.DataFrame(all_results, columns=['sigma', 'lineE', 'norm', 'C-stat_bf', 'new_C-stat', 'delta_C-stat', 'srqt(delta_C-stat)', 'eqwidth (keV)'], dtype=np.float64)
        df_all_results.to_csv(os.path.join(classic_dir, xcmfile.replace('.xcm','') + '_LineSearch.csv'), index = False)
        
        
        # Now Plot everything 
        
        if figure : 
            c = cstat_bf-np.array(cstat)
            c[c<0] = 0
            data_file = xcmfile.replace(".xcm","")+"_LineSearch_plot.npy"
            x,y = np.meshgrid(lineE_list,np.linspace(maxNorm,minNorm,n+1))
            np.save(os.path.join(classic_dir,data_file), np.array([x,y,c],dtype=object))
            x,y,c = np.load(os.path.join(classic_dir,data_file),allow_pickle=True)

            fig,ax = plt.subplots(1,1,figsize=(12,6))
            contour = ax.contourf(x,y,c.T,cmap="Blues")
            plt.xlabel("Energy (keV)")
            plt.ylabel("Norm")
            cbar = plt.colorbar(contour)
            cbar.set_label(r'$\Delta Cstat$')
            plt.hlines(0,minEnergy,maxEnergy,color="k",linestyle="--",alpha=0.5)
            ax.set_xlim(minEnergy,maxEnergy)
            fig.tight_layout()
            
            fig_bis = fig
            fig.savefig(os.path.join(classic_dir, xcmfile.replace(".xcm","")+"_LineSearch_plot.svg"), format="svg",transparent=True)
            fig_bis.savefig(os.path.join(classic_dir, xcmfile.replace(".xcm","")+"_LineSearch_plot.pdf"), format="pdf",transparent=True)


def clear_classic():
    for f in os.listdir(classic_dir):
        os.remove(os.path.join(classic_dir,f))
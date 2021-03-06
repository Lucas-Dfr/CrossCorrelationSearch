from cstat_deviation import *
from xspec import *

"""
This test is meant to compute the cstat deviation for a single file
"""

Xset.restore('f006_TBabs_diskbb_powerlaw_famodel.xcm')

Fit.perform() 
xspec_cstat = Fit.statistic
dof = Fit.dof

# data values should be in counts and not counts/s
t_e = AllData(1).exposure
data_values = []
for i in range (len(AllData(1).values)):
    # Alldata(1).values returns a tuple and i want a list
    data_values.append(AllData(1).values[i]*t_e)
    
# model values in counts and limited to noticed bins
Plot.device = '/null'
Plot.xAxis = 'channel'
Plot('counts')
model_values = Plot.model()

cstat_deviation = db_compute_goodness_of_the_fit_from_cstat_v1(data_values,model_values,dof,xspec_cstat, verbose=True)

print("cstat_deviation:", cstat_deviation)
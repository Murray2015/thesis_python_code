# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:36:54 2017

@author: mxh909
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

log_data = pd.read_csv("Wireline/resolution-1_final.las", delim_whitespace=True, 
                       header=None, names=['DEPTH', 'BS', 'CALI', 'DTC', 'GR', 'GR_CORR', 
                       'RESD', 'RESS', 'SP'], na_values=-999.2500, skiprows=56)

fig = plt.figure(figsize=(8,4))
fig.set_figheight(12)
fig.set_figwidth(10)
#f.title("Resolution-1 Well Log")
ax1 = fig.add_subplot(611)
ax1.set_ylabel('Calliper')
log_data.plot('DEPTH', 'CALI', ax=ax1, color='red', sharex=True,subplots=True, legend=False)
ax2 = fig.add_subplot(612) 
ax2.set_ylabel('DT')
ax2.yaxis.set_label_position("right")
log_data.plot('DEPTH', 'DTC', ax=ax2, color='blue', sharex=True,subplots=True,legend=False)
ax2.yaxis.tick_right()
ax3 = fig.add_subplot(613) 
ax3.set_ylabel('GR')
log_data.plot('DEPTH', 'GR_CORR', color='black', ax=ax3, sharex=True,subplots=True,legend=False)
ax4 = fig.add_subplot(614) 
ax4.set_ylabel('RESD')
ax4.yaxis.set_label_position("right")
log_data.plot('DEPTH', 'RESD', logy=True, color='green', ax=ax4, sharex=True,subplots=True,legend=False)
ax4.yaxis.tick_right()
ax5 = fig.add_subplot(615) 
ax5.set_ylabel('RESS')
log_data.plot('DEPTH', 'RESS', logy=True, color='purple', ax=ax5, sharex=True,subplots=True,legend=False)
ax6 = fig.add_subplot(616) 
ax6.set_ylabel('SP')
ax6.yaxis.set_label_position("right")
log_data.plot('DEPTH', 'SP', color='darkblue', ax=ax6, sharex=True, subplots=True, legend=False)
ax6.yaxis.tick_right()
plt.xlabel("Depth")
plt.savefig("Resolution-1_well_plot.pdf")
plt.show()

# Make a Vp log. For no real reason! 
log_data['VP_m_per_s'] = 1/(log_data['DTC'] * (1e-6) /0.3048)
log_data.plot('DEPTH','VP_m_per_s', legend=False)

### Make time-depth curve by integrating the sonic log 
# Convert from microseconds per ft, to seconds per meter. 
log_data['DT_s_per_m'] = (log_data['DTC'] * (1e-6) * (1/0.3048))

# Fill missing values to enable the cumulative integration 
log_data.loc[0,'DT_s_per_m'] = (1.0/1500)
log_data['DT_s_per_m'] = log_data['DT_s_per_m'].interpolate()

## Cumulativly integrate to form the time depth curve
from scipy.integrate import cumtrapz 
time_s = 2*(cumtrapz(y=log_data['DT_s_per_m'], x=log_data['DEPTH'])) # x2 as log is in one-way-time
time_s += 0.1 # add in the value of the seafloor from the seismic. Entered here rather than calculated, as well is a little upslope.
plt.figure()
plt.plot(time_s, log_data['DEPTH'][:-1])
plt.gca().invert_yaxis()
plt.title("Time-depth curve")
plt.xlabel("Time (s)")
plt.ylabel("Depth (m)")
plt.show()

np.savetxt("Resolution_1_time_depth_curve.txt", np.vstack((log_data['DEPTH'].values[:-1], time_s)).T, header="Depth_m Time_s")

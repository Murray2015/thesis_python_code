# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 12:14:58 2017

@author: murray
"""

import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

log_data = pd.read_csv("Wireline/resolution-1_final.las", delim_whitespace=True, 
                       header=None, names=['DEPTH', 'BS', 'CALI', 'DTC', 'GR', 'GR_CORR', 
                       'RESD', 'RESS', 'SP'], na_values=-999.2500, skiprows=56)

## Fill log data with dummy values for water and sed. Create a dataframe of the filled values, 
## then concat to top of log_data.
water_depth = 64.0 # meters 
first_depth_in_well = log_data['DEPTH'][0]
depths = np.arange(0,first_depth_in_well-1, 1)
DTC_water = np.repeat(1/(1500*3.28084/1e6),water_depth)
DTC_sed = np.repeat(1/(1800*3.28084/1e6),len(depths)-water_depth) # There is a bug here, but it is fixed in lines 63 onwards... 
DTC=np.insert(DTC_sed, 0, DTC_water)
temp_dataframe = pd.DataFrame({'DEPTH':depths, 'DTC':DTC})
log_data_sonic = pd.concat((temp_dataframe,log_data.loc[:,('DTC','DEPTH')]), ignore_index=True)

# Interpolate missing values to enable the cumulative integration 
log_data_sonic.loc[0,'DT_s_per_m'] = (1.0/1500)
log_data_sonic['DT_s_per_m'] = log_data_sonic['DT_s_per_m'].interpolate()
log_data_sonic['DTC'] = log_data_sonic['DTC'].interpolate()

# Make a Vp log, to plot and also to later find the sill velocity from. 
log_data_sonic['VP_m_per_s'] = 1/(log_data_sonic['DTC'] * (1e-6) / 0.3048)
fig = plt.figure(figsize=(4,6))
fig.set_figheight(10)
fig.set_figwidth(7)
ax1 = fig.add_subplot(111)
ax1.add_patch(patches.Rectangle((1500, 1910), 4500, 55, alpha=0.2,facecolor='red')) 
ax1.plot(log_data_sonic['VP_m_per_s'], log_data_sonic['DEPTH'])
ax1.text(4700,1700,r'$ \bar{Vp} = $'+ str(int(log_data_sonic.loc[log_data_sonic['DEPTH']>1911,:]['VP_m_per_s'].mean())), fontsize=14)
ax1.text(4700,1800,r'$ \sigma \ Vp = $'+ str(int(log_data_sonic.loc[log_data_sonic['DEPTH']>1911,:]['VP_m_per_s'].std())), fontsize=14)
plt.gca().invert_yaxis()
plt.xticks(np.arange(1500, 6000, 1000))
plt.xlabel(r'$Vp \quad ms^{-1}$', fontsize=18)
plt.ylabel("Depth (m)", fontsize=14)
#plt.savefig("/Resolution-1_Vp_plot.pdf",bbox_inches='tight')
plt.show()

### Calculate porosity from sonic. Also smooth the sonic with the median filter to remove spikes.
import scipy.signal as ss
log_data_sonic['filtered_sonic'] = ss.medfilt(log_data_sonic['DTC'], kernel_size=15)
plt.plot(log_data_sonic['DTC'], log_data_sonic['DEPTH'], label='unfiltered')
plt.plot(log_data_sonic['filtered_sonic'], log_data_sonic['DEPTH'], label='filtered')
plt.gca().invert_yaxis()
plt.xlabel('DTC')
plt.ylabel('Depth')
plt.legend()
#plt.savefig('filt_Vp.pdf', format='pdf', bbox_inces=20)
plt.show()

Vma = 2700.0 # matrix velocity 1800 to 3500 seem sensible values 
Vl = 1500.0 # fluid velocity 

dt_ma_upper = 127.0 # upper bound of matrix slowness 
dt_ma_lower =  62.0 # lower bound of matrix slowness
dt_f_upper = 189.0 # upper bound of fluid slowness
dt_f_lower = 218.0 # lower bound of fluid slowness

log_data_sonic['Wyllie_porosity_sonic_ll'] = ( (log_data_sonic['filtered_sonic'] - dt_ma_lower) / (dt_f_lower - dt_ma_lower) )
log_data_sonic['Wyllie_porosity_sonic_lu'] = ( (log_data_sonic['filtered_sonic'] - dt_ma_lower) / (dt_f_upper - dt_ma_lower) )
log_data_sonic['Wyllie_porosity_sonic_ul'] = ( (log_data_sonic['filtered_sonic'] - dt_ma_upper) / (dt_f_lower - dt_ma_upper) )
log_data_sonic['Wyllie_porosity_sonic_uu'] = ( (log_data_sonic['filtered_sonic'] - dt_ma_upper) / (dt_f_upper - dt_ma_upper) )

plt.figure(figsize=(6,8))
plt.plot(log_data_sonic['Wyllie_porosity_sonic_ll'], log_data_sonic['DEPTH'], label="ll_w")
plt.plot(log_data_sonic['Wyllie_porosity_sonic_lu'], log_data_sonic['DEPTH'], label="lu_w")
plt.plot(log_data_sonic['Wyllie_porosity_sonic_ul'], log_data_sonic['DEPTH'], label="ul_w")
plt.plot(log_data_sonic['Wyllie_porosity_sonic_uu'], log_data_sonic['DEPTH'], label="uu_w")
plt.gca().invert_yaxis()
plt.xlim(0,1.0)
plt.xlabel(r'$\phi$', fontsize=16)
plt.ylabel('Depth m')
plt.legend(loc=4)
plt.show()

log_data_sonic['Wyllie_porosity_sonic_corr_ll'] = ( ( (log_data_sonic['filtered_sonic'] - dt_ma_lower) / (dt_f_lower - dt_ma_lower) ) * (1.0 / (150.0/100.0)) )
log_data_sonic['Wyllie_porosity_sonic_corr_lu'] =( ( (log_data_sonic['filtered_sonic'] - dt_ma_lower) / (dt_f_upper - dt_ma_lower) ) * (1.0 / (150.0/100.0)) )
log_data_sonic['Wyllie_porosity_sonic_corr_ul'] = ( ( (log_data_sonic['filtered_sonic'] - dt_ma_upper) / (dt_f_lower - dt_ma_upper) ) * (1.0 / (150.0/100.0)) )
log_data_sonic['Wyllie_porosity_sonic_corr_uu'] = ( ( (log_data_sonic['filtered_sonic'] - dt_ma_upper) / (dt_f_upper - dt_ma_upper) ) * (1.0 / (150.0/100.0)) )

plt.figure(figsize=(6,8))
plt.plot(log_data_sonic['Wyllie_porosity_sonic_corr_ll'], log_data_sonic['DEPTH'], label="ll_wc")
plt.plot(log_data_sonic['Wyllie_porosity_sonic_corr_lu'], log_data_sonic['DEPTH'], label="lu_wc")
plt.plot(log_data_sonic['Wyllie_porosity_sonic_corr_ul'], log_data_sonic['DEPTH'], label="ul_wc")
plt.plot(log_data_sonic['Wyllie_porosity_sonic_corr_uu'], log_data_sonic['DEPTH'], label="uu_wc")
plt.gca().invert_yaxis()
plt.xlim(0,1.0)
plt.xlabel(r'$\phi$', fontsize=16)
plt.ylabel('Depth m')
plt.legend(loc=4)
plt.show()

def raymer(dt_ma, dt_l):
    a = (dt_ma / (2.0*dt_l))-1
    return (-a - (((a**2) + (dt_ma / log_data_sonic['filtered_sonic']) -1 ) **(1.0/2.0) ) )
log_data['Raymer_Hunt_Gardner_porosity_sonic_ll'] = raymer(dt_ma_lower, dt_f_lower)
log_data['Raymer_Hunt_Gardner_porosity_sonic_ll'] = raymer(dt_ma_lower, dt_f_upper)
log_data['Raymer_Hunt_Gardner_porosity_sonic_ll'] = raymer(dt_ma_upper, dt_f_lower)
log_data['Raymer_Hunt_Gardner_porosity_sonic_ll'] = raymer(dt_ma_upper, dt_f_upper)
    
plt.figure(figsize=(6,8))
plt.plot(log_data_sonic['Wyllie_porosity_sonic_corr_ll'], log_data_sonic['DEPTH'], label="ll_rhg")
plt.plot(log_data_sonic['Wyllie_porosity_sonic_corr_lu'], log_data_sonic['DEPTH'], label="lu_rhg")
plt.plot(log_data_sonic['Wyllie_porosity_sonic_corr_ul'], log_data_sonic['DEPTH'], label="ul_rhg")
plt.plot(log_data_sonic['Wyllie_porosity_sonic_corr_uu'], log_data_sonic['DEPTH'], label="uu_rhg")
plt.gca().invert_yaxis()
plt.xlim(0,1.0)
plt.xlabel(r'$\phi$', fontsize=16)
plt.ylabel('Depth m')
plt.legend(loc=4)
plt.show()

### What we have found here is that the parameter choices completly dominate the porosity found for whylie time-average. 
### Note I am not using the best params. Read https://www.spec2000.net/12-phidt.htm#b7  for more on
### correcting the porosity log for shale. 
# Equations and parameters from http://petrowiki.org/Porosity_evaluation_with_acoustic_logging 
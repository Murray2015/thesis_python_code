# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:08:42 2017

@author: murray
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

# Read in data 
data = pd.read_csv("opal_ct.csv")
data2 = data[data['Depth_below_seafloor'] < 1000 ]

# Make histogram and cumulative frequency plot 
fig, ax1 = plt.subplots(figsize=(4,5))
ax1.hist(data2['Depth_below_seafloor'], color='b', orientation='horizontal')
ax1.set_ylabel('Depth below seafloor')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_xlabel('Count', color='b')
plt.xlim(0,20)
ax1.tick_params('x', colors='b')
ax2 = ax1.twiny()
ax2.hist(data2['Depth_below_seafloor'], normed=1, histtype='step',cumulative=True, color='r', orientation='horizontal')
ax2.set_xlabel('Cumulative frequency', color='r')
ax2.tick_params('x', colors='r')
plt.gca().invert_yaxis()
fig.tight_layout()
plt.show()

# Make a column of burial rate 
data2['burial_rate'] = data2['Depth_below_seafloor'] / data2['Age_host_seds']

# Plot data on a semilog plot 
plt.scatter(data2['burial_rate'][data2['burial_rate'] < 4], data2['Depth_below_seafloor'][data2['burial_rate'] < 4], color='b') 
plt.scatter(data2['burial_rate'][data2['burial_rate'] > 4], data2['Depth_below_seafloor'][data2['burial_rate'] > 4], color='c') 
plt.scatter(data2['burial_rate'][data2['burial_rate']> 40], data2['Depth_below_seafloor'][data2['burial_rate'] > 40], color='r') 
plt.xlabel('Burial rate (m/Ma)')
plt.ylabel('Depth (m)')
plt.ylim(0,1000)
ax = plt.gca()
ax.invert_yaxis()
ax.set_xscale('log')
plt.show()

# Make cumulative hist for subsets of data
fig, ax1 = plt.subplots(figsize=(4,5))
ax1.hist(data2['Depth_below_seafloor'][data2['burial_rate'] < 4], color='b', orientation='horizontal', bins=range(0,1000,100))
ax1.set_ylabel('Depth below seafloor')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_xlabel('Count', color='b')
ax1.set_xlim(0,15)
ax1.set_ylim([0,1000])
ax1.tick_params('x', colors='b')
ax2 = ax1.twiny()
ax2.hist(data2['Depth_below_seafloor'][data2['burial_rate'] < 4], normed=1, histtype='step',cumulative=True, color='r', orientation='horizontal')
ax2.set_xlabel('Cumulative frequency', color='r')
ax2.tick_params('x', colors='r')
ax2.set_ylim([0,1000])
plt.gca().invert_yaxis()
fig.tight_layout()
plt.show()

fig, ax1 = plt.subplots(figsize=(4,5))
ax1.hist(data2.loc[(data2["burial_rate"] > 4) & (data2["burial_rate"] < 40), "Depth_below_seafloor"], color='c', orientation='horizontal', bins=range(0,1000,100))
ax1.set_ylabel('Depth below seafloor')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_xlabel('Count', color='c')
ax1.set_xlim(0,15)
ax1.set_ylim([0,1000])
ax1.tick_params('x', colors='c')
ax2 = ax1.twiny()
ax2.hist(data2.loc[(data2["burial_rate"] > 4) & (data2["burial_rate"] < 40), "Depth_below_seafloor"], normed=1, histtype='step',cumulative=True, color='r', orientation='horizontal')
ax2.set_xlabel('Cumulative frequency', color='r')
ax2.tick_params('x', colors='r')
ax2.set_ylim([0,1000])
plt.gca().invert_yaxis()
fig.tight_layout()
plt.show()

fig, ax1 = plt.subplots(figsize=(4,5))
ax1.hist(data2['Depth_below_seafloor'][data2['burial_rate'] >40], color='r', orientation='horizontal', bins=range(0,1000,100))
ax1.set_ylabel('Depth below seafloor')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_xlabel('Count', color='r')
ax1.set_xlim(0,15)
ax1.set_ylim([0,1000])
ax1.tick_params('x', colors='r')
ax2 = ax1.twiny()
ax2.hist(data2['Depth_below_seafloor'][data2['burial_rate'] >40], normed=1, histtype='step',cumulative=True, color='r', orientation='horizontal')
ax2.set_xlabel('Cumulative frequency', color='r')
ax2.tick_params('x', colors='r')
ax2.set_ylim([0,1000])
plt.gca().invert_yaxis()
fig.tight_layout()
plt.show()

# 0th order 
## plot cumulative freq against time
# time = depth / burial rate 
# fit regression line, slope is reaction rate k. 

# For inversion, vector of A and vector of E (to make grid search). 
A = np.linspace(start=10e-2, stop=10e4, num=10)
E = np.linspace(start=70, stop=110, num=10)
misfit_grid = np.zeros((10,10))
# find temp at each point with a geothermal gradient of 30 degrees c and a surface temp of 0. 
data2['temp'] = data2['Depth_below_seafloor']/1000 * 30
# Find misfit as abs(predicted - real). Fill it into grid search. 
# Sort by depth 
data3 = data2.sort_values(by='Depth_below_seafloor')
# Make column of cumulative percentage
data3['cum_perc'] = np.cumsum(data3['Depth_below_seafloor'].values)/np.max(np.cumsum(data3['Depth_below_seafloor'].values)) * 100
result = sm.ols(formula="cum_perc ~ Depth_below_seafloor", data=data3).fit()
k = result.params['Depth_below_seafloor']





# 1st order 
## plot ln(cumulative freq) against time
# time = depth / burial rate 
# fit regression line, slope is -ve reaction rate k. 

# For inversion, vector of A and vector of E (to make grid search). 
# find temp at each point with a geothermal gradient of 30 degrees c and a surface temp of 0. 
# Find misfit as abs(predicted - real). Fill it into grid search. 


# Nucleation and growth reaction 
## plot ln(cumulative freq) against time
# time = depth / burial rate 
# fit regression line, slope is -ve reaction rate k. 

# For inversion, vector of A and vector of E (to make grid search). 
# find temp at each point with a geothermal gradient of 30 degrees c and a surface temp of 0. 
# Find misfit as abs(predicted - real). Fill it into grid search. 




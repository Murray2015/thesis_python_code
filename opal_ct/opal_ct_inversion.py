# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:08:42 2017

@author: murray
"""

import pandas as pd 
import matplotlib.pyplot as plt

# Read in data 
data = pd.read_csv("opal_ct.csv")
data2 = data[data['Depth_below_seafloor'] < 1000 ]

# Make histogram and cumulative frequency plot 
fig, ax1 = plt.subplots(figsize=(4,5))
ax1.hist(data2['Depth_below_seafloor'], color='b', orientation='horizontal')
ax1.set_ylabel('Depth below seafloor')
# Make the y-axis label, ticks and tick labels match the line color.
ax1.set_xlabel('Count', color='b')
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
plt.scatter(data2['burial_rate'][data2['burial_rate'] < 3], data2['Depth_below_seafloor'][data2['burial_rate'] < 3], color='b') 
plt.scatter(data2['burial_rate'][data2['burial_rate'] > 3], data2['Depth_below_seafloor'][data2['burial_rate'] > 3], color='c') 
plt.scatter(data2['burial_rate'][data2['burial_rate']> 30], data2['Depth_below_seafloor'][data2['burial_rate'] > 30], color='r') 
plt.xlabel('Burial rate (m/Ma)')
plt.ylabel('Depth (m)')
ax = plt.gca()
ax.invert_yaxis()
ax.set_xscale('log')
plt.show()




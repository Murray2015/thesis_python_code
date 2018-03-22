# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 06:38:17 2017

Wedge model 

This is a seismic wedge model to show that sills are sensibly shown on seismic data in a lateral resolution
sense. 

@author: murray
"""

import numpy as np
import matplotlib.pyplot as plt 
import bruges 
from skimage.util import random_noise as random_noise

## Choose the dimensions of the model, the angle of the wedge, and the vels.
length, depth = 5000, 450 # length based on average sill diam of 10000m, so 5000m radius.
theta = 2.9 # based on arctan(250/5000)
dt = 0.001
rocks = np.array([[1750, 2100],  # Vp, rho
                  [6000, 2900],
                  [1750, 2100]])


## Make a 45 degree wedge 
#model = 1  + np.tri(depth, length, k=-depth//3, dtype=int)
#model[0:depth//3,:] = 0

## Make a not 45 degree wedge 
def make_wedge(width, depth, l1t, theta, dt, noise=False, noise_var=0.1):
    '''
    Make a wedge model given matrix dimensions and wedge angle.
    Makes: 1. Matrix of wedge indecies, 2. Wedge model of velocities and densities, 
    3. Wedge model impedences, 4. Wedge model reflection coefficients all in depth. 
    It then makes a reflection coefficient matrix in time. 
    
    Params:
    width - Width of matrix 
    depth - depth of matrix
    l1t - layer 1 thickness 
    theta - ange of wedge (degrees)
    dt - sample rate of the ricker wavelet 
    
    returns: Depth reflection coefficients,  earth model (vels and rho, width by 
    depth by 2 matrix), list of times, and time reflection coefficients. 
    '''
    model = np.zeros((depth, width))
    theta = np.radians(theta)
    l1t -= 1
    l1t = int(l1t)
    # Make wedge model
    for i in range(depth):
        for j in range(width):
            if i > l1t and (i-l1t) < np.tan(theta)*(j):
                model[i,j] = 1
            elif i > l1t and (i-l1t) >= np.tan(theta)*(j):
                model[i,j] = 2 
    depth_model = model.astype(int)
    # This is very fancy indexing - using the fact that we made the model have integers, we
    # then index into it, where it takes the first, second, or third value in rocks depending
    # on whether the matrix is full of 0, 1 or 2 in the model. 
    earth = rocks[depth_model]
    # Make depth impedence model
    imp = np.apply_along_axis(np.product, -1, earth)
    # Make depth reflection coefficient model
    depth_rc_model = ((imp[1:,:] - imp[:-1,:])) / ((imp[1:,:] + imp[:-1,:]))
    # Find deepest time 
    deepest_time = np.sum(1/earth[:,1,0])*2
    # Make a list of times for the y axis of the time_rc_model matrix
    time_vec = np.arange(0, deepest_time, dt)
    time_rc_model = np.zeros((len(time_vec), width))
    # Find the locations of depth_rc_model in the time grid, and input impedences.
    for i in range(depth-1): # -1 as rc matrix is depth-1,width
        for j in range(width):
            if depth_rc_model[i,j] != 0:
                depth_index = np.argmin(np.abs(np.sum(((2)/earth[0:i,j,0]))-time_vec)) # find in time, then multiply by time step
                time_rc_model[depth_index,j] = depth_rc_model[i,j]
    # If noise is true, add gaussian noise to both reflection coefficient matrixes. 
    if noise:
        depth_rc_noise = random_noise(depth_rc_model, mode='gaussian', clip=False, seed=11, mean=0, var=noise_var)
        depth_rc_model = np.divide(np.add(depth_rc_model, depth_rc_noise),2)
        time_rc_noise = random_noise(time_rc_model, mode='gaussian', clip=False, seed=11, mean=0, var=noise_var)
        time_rc_model = np.divide(np.add(time_rc_model, time_rc_noise),2)
    return depth_rc_model, earth, imp, time_vec, time_rc_model
    
depth_rc_model, earth, imp, time_vec, time_rc_model = make_wedge(width=length,depth=depth,l1t=depth/4,theta=theta, dt=dt, noise=False, noise_var=0.01)
depth_rc_model_n, earth, imp, time_vec, time_rc_model_n = make_wedge(width=length,depth=depth,l1t=depth/4,theta=theta, dt=dt, noise=True, noise_var=0.01)


# Plot depth rc
plt.imshow(depth_rc_model_n, cmap='viridis', aspect='auto')
plt.title("Reflection coefficients: depth")
plt.xlabel('Distance (m)')
plt.ylabel('Depth (m)')
plt.colorbar().ax.set_ylabel('Reflection coefficient')
plt.show()

# Plot time rc
plt.imshow(time_rc_model_n, cmap='viridis', extent=(0, length, np.max(time_vec), 0), aspect='auto')
plt.title("Reflection coefficients: Time")
plt.xlabel('Distance (m)')
plt.ylabel('Two way travel time (s)')
plt.colorbar().ax.set_ylabel('Reflection coefficient')
plt.show()

plt.imshow(earth[:,:,0], cmap='viridis', aspect='auto')
plt.title(r'$V_p$')
plt.xlabel('Distance (m)')
plt.ylabel('Depth (m)')
plt.colorbar().ax.set_ylabel(r'$V_p$ (m/s)')
plt.show()

plt.imshow(earth[:,:,1], cmap='plasma', aspect='auto')
plt.title(r'$\rho$')
plt.xlabel('Distance (m)')
plt.ylabel('Depth (m)')
plt.colorbar().ax.set_ylabel(r'$\rho \/\/ (kg / m^3)$')
plt.show()

# Make a ricker wavelet 
freq = 20
w = bruges.filters.ricker(duration=0.100, dt=dt, f=freq)

# Plots the ricker wavelet 
plt.plot(w)
plt.title('Ricker wavelet, peak frequency=%iHz' %freq)
plt.grid()
plt.show()


##### Plot the clean wedge model ######
# Convolve time reflection coefficient matrix with the ricker wavelet to make synthetic
synth = np.apply_along_axis(lambda t: np.convolve(t, w, mode='same'),
                            axis=0,
                            arr=time_rc_model)
# Plot synthetic seismic
plt.imshow(synth, cmap="seismic", aspect='auto', extent=(0, length, np.max(time_vec), 0))
plt.title('Synthetic seismic wedge model for a sill')
plt.xlabel('Distance (m)')
plt.ylabel('Two way travel time (s)')
plt.colorbar().set_ticks([])
plt.show()

# We can exploit the fact that the reflections have the largest and smallest
# values at the top and bottom of the wedge, to find the amplitudes of the reflections
# at the peak. 
# Find the max of the columns (corresponding to the reflection at the top of the sill)
top_sill = int((((depth/4) / rocks[0,0] ) * 2) / dt)
apparent = np.amax(synth[top_sill-10:top_sill+10,:], axis=0)
actual = synth[top_sill, :]
background = np.average(np.absolute(synth[0:9,:]), axis=0)
import pandas as pd
apparent2 = pd.rolling_mean(apparent, 50)
actual2 = pd.rolling_mean(actual, 50)
background2 = pd.rolling_mean(background, 50)

plt.figure(figsize=(10,7))
plt.plot(apparent2, 'g-', label='Apparent amplitude')
plt.plot(actual2, 'y-', label='Actual amplitude')
plt.plot(background2, 'b-', label='Background average amplitude')
plt.ylabel('Amplitude')
plt.xlabel('Distance (m)')
plt.grid()
plt.ylim(0,np.nanmax(apparent2))
plt.xlim(0,length)
plt.legend()
#plt.savefig('ch2_synthetic_amplitude_curve_clean.jpg', dpi=300)
plt.show()

# Find the locations of the maximum amplitudes for the top surface
apparent_t = np.argmax(synth, axis=0) 
apparent_t = apparent_t.astype(float)
apparent_t[apparent_t==0]=np.nan
actual_t = np.repeat(top_sill, length)

# Plot seismic 
plt.figure(figsize=(10,7))
plt.imshow(synth, aspect='auto', cmap='seismic', extent=(0, length, np.max(time_vec), 0))
plt.colorbar().set_ticks([])
# Plot location of peak amplitude
plt.plot(apparent_t*dt, 'g-', label='Apparent amplitude') # Note *dt. 
plt.plot(actual_t*dt, 'y-', label='Actual amplitude') # Note *dt.
plt.xlabel('Distance (m)')
plt.ylabel('Two way time (s)')
plt.legend()
#plt.savefig('ch2_synthetic_wedge_seismic_clean.jpg', dpi=300)
plt.show()

# Four axes, returned as a 2-d array
f, axarr = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
m1 = axarr[0, 0].imshow(earth[:,:,0], cmap='viridis', aspect='auto')
f.colorbar(m1, ax=axarr[0, 0]).ax.set_ylabel(r'$V_p \/\/ (m/s)$')
axarr[0, 0].set_title(r'$V_p$')
m2 = axarr[0, 1].imshow(earth[:,:,1], cmap='plasma', aspect='auto')
f.colorbar(m2, ax=axarr[0, 1]).ax.set_ylabel(r'$V_p \/\/ (m/s)$')
plt.ylabel('Depth (m)')
axarr[0, 1].set_title(r'$\rho$')
m3 = axarr[1, 0].imshow(imp,  cmap='magma', aspect='auto')
f.colorbar(m3, ax=axarr[1, 0]).ax.set_ylabel(r'$ai \/\/ (Pa. s/m^3)$')
axarr[1, 0].set_title('Acoustic impedance')
m4 = axarr[1, 1].imshow(depth_rc_model, cmap='Greys', aspect='auto')
f.colorbar(m4, ax=axarr[1, 1]).ax.set_ylabel(r'$RC$')
axarr[1, 1].set_title('Reflection coefficient')
# Set common axes labels 
f.text(0.5, 0.04, 'Distance (m)', ha='center')
f.text(0.04, 0.5, 'Depth (m)', va='center', rotation='vertical')
## Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
fig = plt.gcf()
fig.set_size_inches(10, 7)
#plt.savefig('ch2_synthetic_wedge_setup_clean.jpg', dpi=300)
plt.show()

##### Plot the noisy wedge model ######
# Convolve time reflection coefficient matrix with the ricker wavelet to make synthetic
synth_noisy = np.apply_along_axis(lambda t: np.convolve(t, w, mode='same'),
                            axis=0,
                            arr=time_rc_model_n)
# Plot synthetic seismic
plt.imshow(synth_noisy, cmap="seismic", aspect='auto', extent=(0, length, np.max(time_vec), 0))
plt.title('Synthetic seismic wedge model for a sill with noise')
plt.xlabel('Distance (m)')
plt.ylabel('Two way travel time (s)')
plt.colorbar().set_ticks([])
plt.show()

# We can exploit the fact that the reflections have the largest and smallest
# values at the top and bottom of the wedge, to find the amplitudes of the reflections
# at the peak. 
# Find the max of the columns (corresponding to the reflection at the top of the sill)
top_sill = int((((depth/4) / rocks[0,0] ) * 2) / dt)
apparent = np.amax(synth_noisy[top_sill-10:top_sill+10,:], axis=0)
actual = synth_noisy[top_sill, :]
background = np.average(np.absolute(synth_noisy[0:9,:]), axis=0)
import pandas as pd
apparent2 = pd.rolling_mean(apparent, 50)
actual2 = pd.rolling_mean(actual, 50)
background2 = pd.rolling_mean(background, 50)

plt.figure(figsize=(10,7))
plt.plot(apparent2, 'g-', label='Apparent amplitude')
plt.plot(actual2, 'y-', label='Actual amplitude')
plt.plot(background2, 'b-', label='Background average amplitude')
plt.ylabel('Amplitude')
plt.xlabel('Distance (m)')
plt.grid()
plt.ylim(0,np.nanmax(apparent2))
plt.xlim(0,length)
plt.legend()
#plt.savefig('ch2_synthetic_amplitude_curve_noisy.jpg', dpi=300)
plt.show()

# Find the locations of the maximum amplitudes for the top surface
apparent_t = np.argmax(synth_noisy[top_sill-10:top_sill+10,:], axis=0) 
apparent_t += (top_sill-10)
apparent_t = apparent_t.astype(float)
apparent_t[apparent_t==0]=np.nan
actual_t = np.repeat(top_sill, length)

# Plot seismic 
plt.figure(figsize=(10,7))
plt.imshow(synth_noisy, aspect='auto', cmap='seismic', extent=(0, length, np.max(time_vec), 0))
plt.colorbar().set_ticks([])
# Plot location of peak amplitude
plt.plot(apparent_t*dt, 'g-', label='Apparent amplitude') # Note *dt. 
plt.plot(actual_t*dt, 'y-', label='Actual amplitude') # Note *dt.
plt.xlabel('Distance (m)')
plt.ylabel('Two way time (s)')
plt.legend()
#plt.savefig('ch2_synthetic_wedge_seismic_noisy.jpg', dpi=300)
plt.show()
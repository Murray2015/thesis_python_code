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

length, depth = 40, 100
model = 1  + np.tri(depth, length, k=-depth//3, dtype=int)
model[0:depth//3,:] = 0

#plt.imshow(model, cmap='viridis', aspect=0.2)
#plt.show()

rocks = np.array([[2700, 2750],  # Vp, rho
                  [5500, 3250],
                  [2800, 3000]])

# This is very fancy indexing - using the fact that we made the model have integers, we
# then index into it, where it takes the first, second, or third value in rocks depending
# on whether the matrix is full of 0, 1 or 2 in the model.                   
earth = rocks[model]

# Add noise to the model 
noise_amount = 0.1  # multiplied by the variance, so 1 = the var. 
Vp_noise = random_noise(earth[:,:,0], mode='gaussian', clip=False, seed=11, mean=np.mean(earth[:,:,0]), var=np.var(earth[:,:,0])*noise_amount)
earth[:,:,0] = np.divide(np.add(earth[:,:,0], Vp_noise),2)
rho_noise = random_noise(earth[:,:,1], mode='gaussian', clip=False, seed=11, mean=np.mean(earth[:,:,1]), var=np.var(earth[:,:,1])*noise_amount)
earth[:,:,1] = np.divide(np.add(earth[:,:,1], rho_noise),2)


f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
m1 = ax1.imshow(earth[:,:,0], cmap='viridis')
ax1.set_title(r'$V_p$')
f.colorbar(m1, ax=ax1)
m2=ax2.imshow(earth[:,:,1], cmap='plasma')
ax2.set_title(r'$\rho$')
f.colorbar(m2, ax=ax2)
plt.show()

imp = np.apply_along_axis(np.product, -1, earth)
plt.figure()
plt.imshow(imp, aspect=0.2)
plt.colorbar(shrink=0.65)
plt.title('Acoustic impedence')
plt.show()

rc =  (imp[1:,:] - imp[:-1,:]) / (imp[1:,:] + imp[:-1,:])

plt.imshow(rc, cmap='Greys', aspect=0.2)
plt.title('Reflection coefficients')
plt.colorbar(shrink=0.65)
plt.show()

w = bruges.filters.ricker(duration=0.100, dt=0.001, f=40)
plt.plot(w)
plt.title('Ricker wavelet, peak frequency = 40Hz')
plt.show()

synth = np.apply_along_axis(lambda t: np.convolve(t, w, mode='same'),
                            axis=0,
                            arr=rc)

plt.imshow(synth, cmap="Greys", aspect=0.2)
plt.show()

# We can exploit the fact taht the reflections have the largest and smallest
# values at the top and bottom of the wedge, to find the amplitudes of the reflections
# at the peak. 
# Find the max of the columns (corresponding to the reflection at the top of the sill)
apparent = np.amax(synth, axis=0)
actual = synth[depth//3, :]

plt.figure(figsize=(10,7))
plt.plot(apparent, 'r-', label='Apparent amplitude')
plt.plot(actual, 'y-', label='Actual amplitude')
plt.ylabel('Amplitude')
plt.legend()
plt.savefig('ch2_synthetic_amplitude_curve.jpg', dpi=300)
plt.show()

apparent_t = np.argmax(synth, axis=0)
actual_t = np.argmax(rc, axis=0)

# Plot seismic 
plt.figure(figsize=(10,7))
plt.imshow(synth, aspect=0.3, cmap='Greys')
plt.colorbar(shrink=0.925)
# Plot location of peak amplitude
plt.plot(apparent_t, 'r-', label='Apparent amplitude')
plt.plot(actual_t, 'y-', label='Actual amplitude')
plt.legend()
plt.savefig('ch2_synthetic_wedge_seismic.jpg', dpi=300)
plt.show()

# Four axes, returned as a 2-d array
f, axarr = plt.subplots(2, 2)
m1 = axarr[0, 0].imshow(earth[:,:,0], cmap='viridis', aspect=0.3)
f.colorbar(m1, ax=axarr[0, 0])
axarr[0, 0].set_title(r'$V_p$')
m2 = axarr[0, 1].imshow(earth[:,:,1], cmap='plasma', aspect=0.3)
f.colorbar(m2, ax=axarr[0, 1])
axarr[0, 1].set_title(r'$\rho$')
m3 = axarr[1, 0].imshow(imp,  cmap='seismic', aspect=0.3)
f.colorbar(m3, ax=axarr[1, 0])
axarr[1, 0].set_title('Acoustic imp.')
m4 = axarr[1, 1].imshow(rc, cmap='Greys', aspect=0.3)
f.colorbar(m4, ax=axarr[1, 1])
axarr[1, 1].set_title('Reflection coeff.')
# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
fig = plt.gcf()
fig.set_size_inches(10, 7)
plt.savefig('ch2_synthetic_wedge_setup.jpg', dpi=300)
plt.show()

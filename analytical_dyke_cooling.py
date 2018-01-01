# -*- coding: utf-8 -*-
"""
Created on Mon Jan  1 16:39:03 2018

@author: murray

This script solves the anylitical cooling equation for an
intrusion in a constant background temp - it is most appropriate
for a dyke. 

Problems and TODOs: 
diffusion length doesn't work yet. 
Need to check all units and check against published examples. 
unsure why a needs +1 for graphing as vertical line

"""

import numpy as np
from scipy.special import erf as erf
import matplotlib.pyplot as plt

# Initialize params 
T0 = 30.0 # host rock temp, deg C
Tm = 1200.0 # Magma temp, deg C
a = 20.0 # intrusion half width, meters
L = 4.0e5 # latent heat of crystallisation, J/kg per K
cp = 1480.0 # specific heat capacity of intrusion, J/K
k = 1 # thermal conductivity, 
rho = 2830.0 # basalt density, kg/m^3
t = 1.0 # time, in years
grid = np.arange(start=0, stop=100, step=1)

# analytical solution 
def temps(grid=grid, T0=T0, t=t, L=L, cp=cp, Tm=Tm, a=a, k=k):
    Tm += 273.15
    T0 += 273.15
    delta_T = Tm -T0 + (L/cp) 
    K = k / (rho * cp)
    t = t * 365.0 * 24.0 * 60.0 * 60.0 # convert time from years to seconds.
    print("Diffusion length = " + str(np.sqrt(K*t)))
    return (T0 + (delta_T/2.0)*(erf((grid + a)/(2*np.sqrt(K*t))) - erf((grid - a)/(2*np.sqrt(K*t)))  ))-273.15

plt.plot(grid, temps(t=1), 'red')
plt.axvline(x=a, color='k')
plt.text(x=17, y=1300, s="1 year", color='red', size=16)
plt.plot(grid, temps(t=2), 'peru')
plt.text(x=19, y=1150, s="2 years", color='peru', size=16)
plt.plot(grid, temps(t=10), 'orange')
plt.text(x=21, y=1000, s="10 years", color='orange', size=16)
plt.plot(grid, temps(t=20), 'gold')
plt.text(x=23, y=850, s="20 years", color='gold', size=16)
plt.ylabel("Temperature (deg C)")
plt.xlabel("Distance from centre of intrusion (m)")
plt.ylim(0, (Tm + (L/cp)) + 50 )
plt.xlim(np.min(grid), 70)# np.max(grid))
plt.savefig("ch6_dyke_cooling.jpg", dpi=300)
plt.show()

# Sources
# Model: ftp://ftpobs.univ-bpclermont.fr/GEOL/volcano/druitt_esf/5896%20Druitt/Annen.pdf
# Params: http://onlinelibrary.wiley.com/doi/10.1111/bre.12131/pdf
# Params: https://watermark.silverchair.com/egi084.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAbIwggGuBgkqhkiG9w0BBwagggGfMIIBmwIBADCCAZQGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMDHBhggOFdvPzL-5hAgEQgIIBZZHzizF_4_6UJ3VL_enA9h1iywVWEQ7jMqplf8BcOR7QpI_0Z0p_xbgfDAypTZ_1q69BFJeeggkWuYXzas-Hk3eR25BzbP_0A35GD45osfUqxXBWLdYLnQtgKrpXGMKoeOYkCGceiGNMJVCMivefVXg0jPZraX7vP4kkQOCq1kYhvq8ISfvxCIFkPF1Ungq0qdCONH6xZpGtFVkzfK0vFDjA5AJ71f5SBhXhi0-O2Z2_uWA_FYDBMq3y1vmXNsFFhAeOik0rDlfbvPqtQIBFb8qVk5g_x9TqXFFCnk2NMs4ZpeiG3CTSyVUvICMFH8zgV8mZ-mLMNLbNW8E2xZR5ImtIEKWMInSSZJposImjNo2ngk1DhDQQxCGFFY8Lb11cfqC9OCxoKX1JtsDisebzGC5rUSY5LyxoKIAFHESYB_BkDgbRm6y8UrO0T3HbBoFMo330Mc6ynPEYVCdT3jRjJ3GviDS0wQ

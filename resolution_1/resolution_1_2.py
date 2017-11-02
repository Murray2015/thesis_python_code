# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:36:54 2017

@author: mxh909
"""

import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

log_data = pd.read_csv("Wireline/resolution-1_final.las", delim_whitespace=True, 
                       header=None, names=['DEPTH', 'BS', 'CALI', 'DTC', 'GR', 'GR_CORR', 
                       'RESD', 'RESS', 'SP'], na_values=-999.2500, skiprows=56)

fig = plt.figure(figsize=(8,4))
fig.set_figheight(12)
fig.set_figwidth(10)
ax1 = fig.add_subplot(511)
ax1.set_ylabel('Calliper')
log_data.plot('DEPTH', 'CALI', ax=ax1, color='red', sharex=True,subplots=True, legend=False)
ax2 = fig.add_subplot(512) 
ax2.set_ylabel('DT')
ax2.yaxis.set_label_position("right")
log_data.plot('DEPTH', 'DTC', ax=ax2, color='blue', sharex=True,subplots=True,legend=False)
ax2.yaxis.tick_right()
ax3 = fig.add_subplot(513) 
ax3.set_ylabel('GR')
log_data.plot('DEPTH', 'GR_CORR', color='black', ax=ax3, sharex=True,subplots=True,legend=False)
ax4 = fig.add_subplot(514) 
log_data.plot('DEPTH', 'RESS', logy=True, color='purple', ax=ax4, sharex=True,subplots=True)
log_data.plot('DEPTH', 'RESD', logy=True, color='green', ax=ax4, sharex=True,subplots=True)
ax4.yaxis.tick_right()
ax4.yaxis.set_label_position("right")
ax4.set_ylabel('Resistivity')
ax5 = fig.add_subplot(515) 
ax5.set_ylabel('SP')
log_data.plot('DEPTH', 'SP', color='darkblue', ax=ax5, sharex=True, subplots=True, legend=False)
plt.xlabel("Depth (m)")
#plt.savefig("/home/mxh909/Desktop/magee_resolution_images/Resolution-1_well_plot.pdf")
plt.show()

## Fill log data with dummy values for water and sed. Create a dataframe of the filled values, 
## then concat to top of log_data.
water_depth = 64.0 # meters 
first_depth_in_well = log_data['DEPTH'][0]
depths = np.arange(0,first_depth_in_well-1, 1)
BS = np.repeat(10,len(depths))
CALI= np.repeat(np.nan,len(depths))
DTC_water = np.repeat(1/(1500*3.28084/1e6),water_depth)
DTC_sed = np.repeat(1/(1800*3.28084/1e6),len(depths)-water_depth) # There is a bug here, but it is fixed in lines 63 onwards... 
DTC=np.insert(DTC_sed, 0, DTC_water)
GR= np.repeat(np.nan,len(depths))
GR_CORR= np.repeat(np.nan,len(depths))
RESD= np.repeat(np.nan,len(depths))
RESS= np.repeat(np.nan,len(depths))
SP= np.repeat(np.nan,len(depths))
temp_dataframe = pd.DataFrame({'DEPTH':depths, 'BS':BS, 'CALI':CALI, 'DTC':DTC, 'GR':GR, 'GR_CORR':GR_CORR, 'RESD':RESD, 'RESS':RESS, 'SP':SP})
log_data = pd.concat((temp_dataframe,log_data), ignore_index=True)

# Interpolate missing values to enable the cumulative integration 
log_data.loc[0,'DT_s_per_m'] = (1.0/1500)
log_data['DT_s_per_m'] = log_data['DT_s_per_m'].interpolate()
log_data['DTC'] = log_data['DTC'].interpolate()

# Make a Vp log, to plot and also to later find the sill velocity from. 
log_data['VP_m_per_s'] = 1/(log_data['DTC'] * (1e-6) / 0.3048)
fig = plt.figure(figsize=(4,6))
fig.set_figheight(10)
fig.set_figwidth(7)
ax1 = fig.add_subplot(111)
ax1.add_patch(patches.Rectangle((1500, 1910), 4500, 55, alpha=0.2,facecolor='red')) 
ax1.plot(log_data['VP_m_per_s'], log_data['DEPTH'])
ax1.text(4700,1700,r'$ \bar{Vp} = $'+ str(int(log_data.loc[log_data['DEPTH']>1911,:]['VP_m_per_s'].mean())), fontsize=14)
ax1.text(4700,1800,r'$ \sigma \ Vp = $'+ str(int(log_data.loc[log_data['DEPTH']>1911,:]['VP_m_per_s'].std())), fontsize=14)
plt.gca().invert_yaxis()
plt.xticks(np.arange(1500, 6000, 1000))
plt.xlabel(r'$Vp \quad ms^{-1}$', fontsize=18)
plt.ylabel("Depth (m)", fontsize=14)
plt.savefig("/home/mxh909/Desktop/magee_resolution_images/Resolution-1_Vp_plot.pdf",bbox_inches='tight')
plt.show()


### Make time-depth curve by integrating the sonic log 
# Convert from microseconds per ft, to seconds per meter. 
log_data['DT_s_per_m'] = (log_data['DTC'] * (1e-6) * (1 / 0.3048))
# Cumulativly integrate to form the time depth curve
from scipy.integrate import cumtrapz 
time_s = 2*(cumtrapz(y=log_data['DT_s_per_m'], x=log_data['DEPTH'])) # x2 as log is in one-way-time
time_s += 0.1 # add in the value of the seafloor from the seismic. Entered here rather than calculated, as well is a little upslope.
plt.figure()
plt.plot(time_s, log_data['DEPTH'][:-1])
plt.gca().invert_yaxis()
plt.title("Time-depth curve")
plt.xlabel("Time (s)")
plt.ylabel("Depth (m)")
plt.savefig("/home/mxh909/Desktop/magee_resolution_images/Resolution-TD_curve.pdf",bbox_inches='tight')
plt.show()

np.savetxt("/home/mxh909/Desktop/magee_resolution_images/Resolution_1_time_depth_curve.txt", np.vstack((log_data['DEPTH'].values[:-1], time_s)).T, header="Depth_m Time_s")

### Calculate porosity from sonic. Also smooth the snoic with the median filter to remove spikes.
import scipy.signal as ss
log_data['filtered_VP_m_per_s'] = ss.medfilt(log_data['VP_m_per_s'], kernel_size=3)
#plt.plot(log_data['VP_m_per_s'], log_data['DEPTH'], label='unfiltered')
#plt.plot(log_data['filtered_VP_m_per_s'], log_data['DEPTH'], label='filtered')
#plt.legend()
#plt.savefig('filt_Vp.pdf', format='pdf', bbox_inces=20)
#plt.show()

Vma = 3400.0 # matrix velocity 
Vl = 1500.0 # fluid velocity 
log_data['Wyllie_porosity_sonic'] = ((1/log_data['filtered_VP_m_per_s'])-(1.0/Vma))/((1.0/Vl)-(1.0/Vma)) 
log_data['Wyllie_porosity_sonic_shale'] = ((1/log_data['filtered_VP_m_per_s'])-(1.0/Vma))/((1.0/Vl)-(1.0/Vma)) * 0.75 # but I can't work out this correction factor legitamately
log_data['Raymer_Hunt_Gardner_porosity_sonic'] = 0.67*((1/log_data['filtered_VP_m_per_s'])-(1.0/Vma))/((1.0/log_data['filtered_VP_m_per_s']))

plt.figure(figsize=(6,8))
plt.plot(log_data['Raymer_Hunt_Gardner_porosity_sonic'], log_data['DEPTH'], label="RHG")
plt.plot(log_data['Wyllie_porosity_sonic'], log_data['DEPTH'], label="Wyllie time-average")
plt.plot(log_data['Wyllie_porosity_sonic_shale'], log_data['DEPTH'], label="Wyllie time-average_corr")
plt.gca().invert_yaxis()
plt.xlim(0,1.0)
plt.xlabel(r'$\phi$', fontsize=16)
plt.ylabel('Depth m')
plt.legend(loc=4)
plt.show()

#### This is possibly done. We have found the resistivity logs to be useless, as we havn't used
#### real parameters for it. Maybe this will work, but will need to wade through well reports finding
#### parameters. RHG seems to be the most legit, although deciding on what the matrix velocity is 
#### is tricky. Next up: use my forced fold code to make an optimization model to fit the theoretical
#### porosity curve with. 

# Make density log with Gardners law 
log_data['density_log_kg_m3'] = ( 0.31 * log_data['filtered_VP_m_per_s']**0.25) * 1e3

# Make effective_stress by integrating the density log 
log_data['effective_stress_MPa'] = np.insert((cumtrapz(y=log_data['density_log_kg_m3'], x=log_data['DEPTH']) * 1e-5), 0, 0)
log_data['effective_stress_MPa'].plot()

### Find best fit porosity, restricting the values to fit between the top of the real data,
### and the top of the sill.
#from scipy.optimize import minimize
#def misfit(par, log_data): 
#    # Fit theoretical model 
#    theoretical_poro = par[0] * np.exp(-par[1]*log_data['effective_stress_MPa'][2145:12100])    
#    mse = np.sum(np.power(theoretical_poro - log_data['Raymer_Hunt_Gardner_porosity_sonic'][2145:12100], 2))
#ion#    return mse 
#    
#par = [0.5,0.03]
#best_par = minimize(misfit, x0=par, args=(log_data), bounds=[(0.0,1.0),(0,2)])

######## This is the ammended version of above, which has been altered to follow the equation phi = phi0 * e ** -c*z, where z is depth and c is a coefficient. 
## Find best fit porosity, restricting the values to fit between the top of the real data,
## and the top of the sill.
from scipy.optimize import minimize
def misfit(par, log_data=log_data): 
    # Fit theoretical model 
    theoretical_poro = par[0] * np.exp(-par[1]*(log_data['DEPTH'][2145:12100]/1000))    
    mse = np.sum(np.power(theoretical_poro - log_data['Raymer_Hunt_Gardner_porosity_sonic'][2145:12100], 2))
    return mse 
    
par = [0.5,0.45] # par0 = between 0.3 and 0.7 (ie it is phi0). par1 = c = betweek 0.27 and 0.71. 
best_par = minimize(misfit, x0=par, args=(log_data), bounds=[(0.4,0.75),(0.25,0.65)])


## Plot the best porosity and the Raymer Hunt Gardner porosity
best_theoretical_poro = best_par['x'][0] * np.exp(-best_par['x'][1]*log_data['DEPTH']/1000)    
plt.figure(figsize=(6,8))
plt.plot(log_data['Raymer_Hunt_Gardner_porosity_sonic'], log_data['DEPTH'], label="RHG")
plt.plot(best_theoretical_poro, log_data['DEPTH'], label='Theoretical')
plt.gca().invert_yaxis()
plt.xlim(0,1.0)
plt.xlabel(r'$\phi$', fontsize=16)
plt.ylabel('Depth m')
plt.text(0.45,200,r'$ \phi_0 = $'+ str(round(best_par['x'][0], 3)), fontsize=14)
plt.text(0.45,300,r'$ c = $'+ str(round(best_par['x'][1], 3)), fontsize=14)
plt.legend(loc=4)
plt.savefig("/home/mxh909/Desktop/magee_resolution_images/Resolution-calculated_and_theoretical_poro.pdf",bbox_inches='tight')
plt.show()



#### Try grid search for backstripping parameters ####
from scipy.optimize import brute 
def misfit2(par, log_data=log_data): 
    # Fit theoretical model 
    theoretical_poro = par[0] * np.exp(-par[1]*(log_data['DEPTH'][2145:12100]/1000))    
    mse = np.sum(np.power(theoretical_poro - log_data['Raymer_Hunt_Gardner_porosity_sonic'][214x05:12100], 2))
    return mse 

# First slice in ranges is phi0, second is c. 
x0, fval, grid, jout = brute(func=misfit2, ranges=(slice(0.3,0.75,0.01),slice(0.25,0.65, 0.01)), 
                              Ns=20, finish=None, full_output=True)

# The extents of the grid is confusing... but I'm fairly sure this is right. 
plt.imshow(jout, extent=[0.75, 0.3, 0.25, 0.65])
plt.xlabel(r'$\phi_0$', fontsize=16)
plt.ylabel('C')
plt.title('MSE of decompaction parameters')
plt.colorbar()
plt.savefig('/home/mxh909/Desktop/magee_resolution_images/Resolution-decomp_grid_search.pdf',bbox_inches='tight')

#########################################
############### Load files ##############
#########################################
## Read in horizons exported from kingdom. 
def read_horizon_files(file):
    assert type(file) == str, "File not a string"
    return pd.read_csv(file, sep='\s+', header=None, names=['x_utm','y_utm','line','trace','depth','amplitude'], engine='python')


sea_floor = read_horizon_files('~/Documents/random_work/resolution-1-magee/Resolution-1-well/resolution_depth_horizons_23-5-2017/sea_floor.dat')
taranaki = read_horizon_files('~/Documents/random_work/resolution-1-magee/Resolution-1-well/resolution_depth_horizons_23-5-2017/taranaki.dat')
top_forced_fold = read_horizon_files('~/Documents/random_work/resolution-1-magee/Resolution-1-well/resolution_depth_horizons_23-5-2017/top_forced_fold.dat')
southland = read_horizon_files('~/Documents/random_work/resolution-1-magee/Resolution-1-well/resolution_depth_horizons_23-5-2017/southland.dat')
pareora = read_horizon_files('~/Documents/random_work/resolution-1-magee/Resolution-1-well/resolution_depth_horizons_23-5-2017/pareora.dat')
landon = read_horizon_files('~/Documents/random_work/resolution-1-magee/Resolution-1-well/resolution_depth_horizons_23-5-2017/landon.dat')
arnold = read_horizon_files('~/Documents/random_work/resolution-1-magee/Resolution-1-well/resolution_depth_horizons_23-5-2017/arnold.dat')
dannevirke = read_horizon_files('~/Documents/random_work/resolution-1-magee/Resolution-1-well/resolution_depth_horizons_23-5-2017/dannevirke.dat')
mata = read_horizon_files('~/Documents/random_work/resolution-1-magee/Resolution-1-well/resolution_depth_horizons_23-5-2017/mata.dat')
top_sill = read_horizon_files('~/Documents/random_work/resolution-1-magee/Resolution-1-well/resolution_depth_horizons_23-5-2017/top_sill.dat')
base_sill = read_horizon_files('~/Documents/random_work/resolution-1-magee/Resolution-1-well/resolution_depth_horizons_23-5-2017/base_sill.dat')

top_sill_time = read_horizon_files('~/Documents/random_work/resolution-1-magee/Resolution-1-well/resolution_depth_horizons_23-5-2017/sill_time_grids/top_sill_time.dat')
base_sill_time = read_horizon_files('~/Documents/random_work/resolution-1-magee/Resolution-1-well/resolution_depth_horizons_23-5-2017/sill_time_grids/base_sill_time.dat')
base_sill_time2 = read_horizon_files('~/Documents/random_work/resolution-1-magee/Resolution-1-well/resolution_depth_horizons_23-5-2017/sill_time_grids/base_sill_time_2.dat')
top_sill_time = top_sill_time.rename(columns = {'depth':'time'})
base_sill_time = base_sill_time.rename(columns = {'depth':'time'})
base_sill_time2 = base_sill_time2.rename(columns = {'depth':'time'})
arnold2 = read_horizon_files('~/Documents/random_work/resolution-1-magee/Resolution-1-well/resolution_depth_horizons_23-5-2017/arnold_preclip2.dat')


##################################
############## Clip ##############  # Note think if this is needed for fold - probably needs padding either side,
##################################    otherwise it is a bug. 
## Sill is already clipped

def horizon_clip(horizon, sill):
    '''
    Function to extract a horizon above a sill. It DOES NOT
    subtract the background trend to find the actual amplitude. 

    '''
    return horizon[horizon['trace'].isin(sill['trace'])]

dannevirke_clip = horizon_clip(dannevirke, base_sill).reset_index(drop=True)
sea_floor_clip = horizon_clip(sea_floor, dannevirke_clip).reset_index(drop=True)
taranaki_clip = horizon_clip(taranaki, dannevirke_clip).reset_index(drop=True)
top_forced_fold_clip = horizon_clip(top_forced_fold, dannevirke_clip).reset_index(drop=True)
southland_clip = horizon_clip(southland, dannevirke_clip).reset_index(drop=True)
pareora_clip = horizon_clip(pareora, dannevirke_clip).reset_index(drop=True)
landon_clip = horizon_clip(landon, dannevirke_clip).reset_index(drop=True)
arnold_clip = horizon_clip(arnold, dannevirke_clip).reset_index(drop=True)
mata_clip = horizon_clip(mata, dannevirke_clip).reset_index(drop=True)
top_sill_clip = horizon_clip(top_sill, dannevirke_clip).reset_index(drop=True)
base_sill_clip = horizon_clip(base_sill, dannevirke_clip).reset_index(drop=True)

top_sill_time= horizon_clip(top_sill_time, dannevirke_clip).reset_index(drop=True)
base_sill_time= horizon_clip(base_sill_time, dannevirke_clip).reset_index(drop=True)
base_sill_time2= horizon_clip(base_sill_time2, dannevirke_clip).reset_index(drop=True)

for horizon in [sea_floor_clip, taranaki_clip, top_forced_fold_clip, southland_clip ,pareora_clip ,landon_clip ,arnold_clip ,dannevirke_clip ,mata_clip ,top_sill_clip ,base_sill_clip ]:
    print(len(horizon))
    
## Find which trace is missing in taranaki_clip, and drop it from the other dataframes
for horizon in [sea_floor_clip, top_forced_fold_clip, southland_clip ,pareora_clip ,landon_clip ,arnold_clip ,dannevirke_clip ,mata_clip ,top_sill_clip ,base_sill_clip ]:
    for i in horizon['trace'].values:
        if i not in taranaki_clip['trace'].values:
            print i
## This shows that trace 730.0 

## Remove trace 730 from all files and reset indexes
sea_floor_clip = sea_floor_clip.drop(sea_floor_clip[sea_floor_clip.trace == 730.0].index)
sea_floor_clip = sea_floor_clip.reset_index(drop=True)
top_forced_fold_clip = top_forced_fold_clip.drop(top_forced_fold_clip[top_forced_fold_clip.trace == 730.0].index)
top_forced_fold_clip = top_forced_fold_clip.reset_index(drop=True)
southland_clip = southland_clip.drop(southland_clip[southland_clip.trace == 730.0].index)
southland_clip = southland_clip.reset_index(drop=True)
pareora_clip = pareora_clip.drop(pareora_clip[pareora_clip.trace == 730.0].index)
pareora_clip = pareora_clip.reset_index(drop=True)
landon_clip = landon_clip.drop(landon_clip[landon_clip.trace == 730.0].index)
landon_clip = landon_clip.reset_index(drop=True)
arnold_clip = arnold_clip.drop(arnold_clip[arnold_clip.trace == 730.0].index)
arnold_clip = arnold_clip.reset_index(drop=True)
dannevirke_clip = dannevirke_clip.drop(dannevirke_clip[dannevirke_clip.trace == 730.0].index)
dannevirke_clip = dannevirke_clip.reset_index(drop=True)
mata_clip = mata_clip.drop(mata_clip[mata_clip.trace == 730.0].index)
mata_clip = mata_clip.reset_index(drop=True)
top_sill_clip = top_sill_clip.drop(top_sill_clip[top_sill_clip.trace == 730.0].index)
top_sill_clip = top_sill_clip.reset_index(drop=True)
base_sill_clip = base_sill_clip.drop(base_sill_clip[base_sill_clip.trace == 730.0].index)
base_sill_clip = base_sill_clip.reset_index(drop=True)

### Check the above has worked
for horizon in [sea_floor_clip, taranaki_clip, top_forced_fold_clip, southland_clip ,pareora_clip ,landon_clip ,arnold_clip ,dannevirke_clip ,mata_clip ,top_sill_clip ,base_sill_clip ]:
    print(len(horizon))
## This shows it has worked
    




#######################################
######### Find Sill Thickness #########
#######################################
def depth_sill_amp_extraction_2d(top,base):
    '''
    Function to test for corresponding points in sill top and base. If found to 
    correspond, the function subtracts the top from the base to give the sill 
    amplitude in meters. Returns a numpy array of amplitude in twtt (ms) 
    in row index 0, x_coordinate in row index 1, and line name in row index 2. 
    '''
    sill_amp = []
    x_coord = []
    line = []
    trace = []
    for i in range(len(top['x_utm'])):
        for j in range(len(base['x_utm'])):
            if top['x_utm'][i] == base['x_utm'][j]:
                if top['line'][i] == base['line'][j]:
                    sill_amp.append(float(base['depth'][j])  -  float(top['depth'][i]))
                    x_coord.append(float(top['x_utm'][i]))
                    line.append(str(top['line'][i]))
                    trace.append(float(top['trace'][i]))
    return pd.DataFrame({"sill_amp_meters":sill_amp, 'x_utm':x_coord, 'line':line, 'trace':trace})

sill_amp_from_depth = depth_sill_amp_extraction_2d(top_sill_clip, base_sill_clip)



def time_sill_amp_extraction_2d(top,base,sill_vel):
    '''
    Function to test for corresponding points in sill top and base. If found to 
    correspond, the function subtracts the top from the base to give the sill 
    amplitude in milliseconds. Returns a numpy array of amplitude in twtt (ms) 
    in row index 0, x_coordinate in row index 1, and line name in row index 2. 
    '''
    sill_amp_time = []
    x_coord = []
    line = []
    trace = []
    sill_amp_depth = []
    for i in range(len(top['x_utm'])):
        for j in range(len(base['x_utm'])):
            if top['x_utm'][i] == base['x_utm'][j]:
                if top['line'][i] == base['line'][j]:
                    sill_amp_time.append(float(base['time'][j])  -  float(top['time'][i]))
                    x_coord.append(float(top['x_utm'][i]))
                    line.append(str(top['line'][i]))
                    trace.append(float(top['trace'][i]))
                    sill_amp_depth.append(((float(base['time'][j])  -  float(top['time'][i]))/2) * sill_vel)
    return pd.DataFrame({"sill_amp_meters":sill_amp_depth, "sill_amp_time":sill_amp_time, 'x_utm':x_coord, 'line':line, 'trace':trace})

sill_amp_from_time = time_sill_amp_extraction_2d(top_sill_time, base_sill_time, sill_vel=5000)
sill_amp_from_time2 = time_sill_amp_extraction_2d(top_sill_time, base_sill_time2, sill_vel=5000)


#sill_amp_from_time.plot("x_utm", "sill_amp_meters")

## Plot all sill thicknesses in time to check they look sensible. 
def make_sill_thickness_plot(sill):
    '''
    This function takes a pandas dataframe of sill thicknesses and plots
    them against the x coordinate. No other transforms are applied. 
    '''
    plt.plot(sill['x_utm'], sill['sill_amp_meters'])
    plt.ylabel('Sill thickness (meters)')
    plt.xlabel('x coordinate (UTM 29N)')
    plt.show()
    
make_sill_thickness_plot(sill_amp)

##################################
########## Decompact #############
##################################
def decompact_fold(fold, z3=0, sediment_density=2000, tolerance=0.000001, phi0=0.6, Lambda=2):
    '''
    This function decompacts a forced fold by repeatedly calling the function decompact
    '''
    z4_output = []   
    
    def decompact(z1, z2, z3, sediment_density, tolerance, phi0, Lambda):
        '''
        This function takes the depth at the top of a unit (z1) and the base (z2), and
        finds what depth z2 will be at if all strata above z1 are removed, such that
        z1 moves to z3 depth. This defaults to 0, but can be changed. the new depth 
        of z2 is called z4. Note z1,z2 and z3 must be give in kilometers. 
        '''
        z4_old = (z1+z2)+2/2
        z4_new = (z1+z2)/2
        
        while abs(z4_old-z4_new) > tolerance:
            z4_old = z4_new
            z4_new = z2 - z1 + phi0*Lambda*(np.exp(-z2/Lambda)-np.exp(-z1/Lambda)+1-np.exp(-z4_old/Lambda))        
        return z4_new
    
    for i in range(len(fold)):
        z4_output.append(decompact(fold['depth'].iloc[i]/1000, fold['depth'].max()/1000, z3, sediment_density, tolerance, phi0, Lambda))    
    fold['decompacted_profile_km'] = pd.Series(z4_output, index=fold.index)
    return fold 

fold_decompacted = decompact_fold(top_forced_fold_clip, sediment_density=2000, tolerance=0.0001, phi0=0.6, Lambda=2)
arnold_decompacted = decompact_fold(arnold_clip, sediment_density=2000, tolerance=0.0001, phi0=0.6, Lambda=2)
arnold2_decompacted = decompact_fold(arnold2, sediment_density=2000, tolerance=0.0001, phi0=0.6, Lambda=2)

##################################
############ Detrend #############
##################################
def detrend_fold_profiles(fold):
    '''
    This function detrends the fold profile to remove the effect of being on a slope.
    We fit a linear trend between the first and last points in the file, and then
    subtract that trend from the points in the file. This is nor fantastically resistant
    to the edges of the fold curving up due to mispicking, but it works. 
    '''
    fold2 = fold.copy()
    ## Detrend deompacted fold profile
    first_point = (fold2['x_utm'].iloc[0], fold2['decompacted_profile_km'].iloc[0])
    last_point = (fold2['x_utm'].iloc[-1], fold2['decompacted_profile_km'].iloc[-1])
    m = (last_point[1] - first_point[1])/(last_point[0]-first_point[0])
    b = first_point[1] - m*(first_point[0])
    fold2['decompacted_profile_detrended_km'] = fold2['decompacted_profile_km']
    fold2['decompacted_profile_detrended_km'] -= (fold2['x_utm']*m + b)
    fold2['decompacted_profile_detrended_km'] = abs(fold2['decompacted_profile_detrended_km']) 
    ## Detrend standard fold profile
    first_point2 = (fold2['x_utm'].iloc[0], fold2['depth'].iloc[0])
    last_point2 = (fold2['x_utm'].iloc[-1], fold2['depth'].iloc[-1])
    m = (last_point2[1] - first_point2[1])/(last_point2[0]-first_point2[0])
    b = first_point2[1] - m*(first_point2[0])
    fold2['fold_profile_detrended'] = fold2['depth']
    fold2['fold_profile_detrended'] -= (fold2['x_utm']*m + b)
    fold2['fold_profile_detrended'] = abs(fold2['fold_profile_detrended']) 
    return fold2

fold_detrended = detrend_fold_profiles(fold_decompacted)
arnold_detrended = detrend_fold_profiles(arnold_decompacted)
arnold2_detrended = detrend_fold_profiles(arnold2_decompacted)


## Plot to check has worked ok. 
def plot_sills_and_folds(fold, sill):
    plt.plot(fold['x_utm'], fold['decompacted_profile_detrended_km']*1000, label='Decompacted fold')
    plt.plot(fold['x_utm'], fold['fold_profile_detrended'], label='Fold')
    plt.plot(sill['x_utm'], sill['sill_amp_meters'], label='Sill amplitude')
    plt.legend()
    plt.show()

plot_sills_and_folds(fold_detrended, sill_amp)
plot_sills_and_folds(fold_detrended, sill_amp_from_time)
plot_sills_and_folds(fold_detrended, sill_amp_from_time2)

## Full flow - note wrong as cutting part way up fold 
sill_amp_from_time2 = time_sill_amp_extraction_2d(top_sill_time, base_sill_time2, sill_vel=5000)
fold_decompacted = decompact_fold(top_forced_fold_clip, sediment_density=2000, tolerance=0.00000001, phi0=0.4, Lambda=1.54)
fold_detrended = detrend_fold_profiles(fold_decompacted)
plot_sills_and_folds(fold_detrended, sill_amp_from_time2)

## FUll flow 2 - with arnold not top forced fold - note wrong as cutting part way up fold 
sill_amp_from_time2 = time_sill_amp_extraction_2d(top_sill_time, base_sill_time2, sill_vel=5000)
arnold_decompacted = decompact_fold(arnold_clip, sediment_density=2000, tolerance=0.00000001, phi0=0.5, Lambda=1.54)
arnold_detrended = detrend_fold_profiles(arnold_decompacted)
plot_sills_and_folds(arnold_detrended, sill_amp_from_time2)

## FUll flow 2 - with new arnold not top forced fold
# Find mean and standard deviation of VP
sill_mean = log_data.loc[log_data['DEPTH']>1911,:]['VP_m_per_s'].mean()
sill_std = log_data.loc[log_data['DEPTH']>1911,:]['VP_m_per_s'].std()
sill_amp_from_time2_mean = time_sill_amp_extraction_2d(top_sill_time, base_sill_time2, sill_vel=sill_mean)
sill_amp_from_time2_upper = time_sill_amp_extraction_2d(top_sill_time, base_sill_time2, sill_vel=sill_mean+(2*sill_std))
sill_amp_from_time2_lower = time_sill_amp_extraction_2d(top_sill_time, base_sill_time2, sill_vel=sill_mean-(2*sill_std))
arnold2_decompacted = decompact_fold(arnold2, sediment_density=2000, tolerance=0.00000001, phi0=0.4, Lambda=1.54)
arnold2_detrended = detrend_fold_profiles(arnold2_decompacted)
# Plot the data. Note this code is dirty - sill_amp_from_time2_lower and the others all have the same name column, even though it contains differen things...
plt.plot(arnold2_detrended['x_utm'], arnold2_detrended['decompacted_profile_detrended_km']*1000, label='Decompacted fold')
plt.plot(arnold2_detrended['x_utm'], arnold2_detrended['fold_profile_detrended'], label='Fold')
plt.plot(sill_amp_from_time2['x_utm'], sill_amp_from_time2['sill_amp_meters'], label='Sill amplitude')
plt.fill_between(sill_amp_from_time2['x_utm'], sill_amp_from_time2_lower['sill_amp_meters'], sill_amp_from_time2_upper['sill_amp_meters'], color='b', alpha=0.2, )
plt.ylabel("Sill thickness / Fold amplitude (meters)")
plt.xlabel("x utm")
plt.legend(loc=8)
plt.savefig("/home/mxh909/Desktop/magee_resolution_images/Resolution-decomp_fold.pdf",bbox_inches='tight')
plt.show()



# Sensible values of lambda: 1.4 to 3.7
## Things to try: 1. try on Arnold, given that possible slight erosion at top of forced fold. 
## 3. Make sure am clipping based on the TOP of the sill not the base, as the base doesn't cover as great of a lateral area. 

# best params phi0 = 0.4 ,  c=0.65 (note c = 1/lambda, therefore lambda = 1.54)
# sill depth in well: 1911m to 1963m

## FUll flow 3 - with new arnold not top forced fold
# Find mean and standard deviation of VP
sill_mean = log_data.loc[log_data['DEPTH']>1911,:]['VP_m_per_s'].mean()
sill_std = log_data.loc[log_data['DEPTH']>1911,:]['VP_m_per_s'].std()
sill_amp_from_time2_mean = time_sill_amp_extraction_2d(top_sill_time, base_sill_time2, sill_vel=sill_mean)
sill_amp_from_time2_upper = time_sill_amp_extraction_2d(top_sill_time, base_sill_time2, sill_vel=sill_mean+(2*sill_std))
sill_amp_from_time2_lower = time_sill_amp_extraction_2d(top_sill_time, base_sill_time2, sill_vel=sill_mean-(2*sill_std))
## Make upper bound with best params 
arnold2_decompacted = decompact_fold(arnold2, sediment_density=2000, tolerance=0.00000001, phi0=0.35, Lambda=1.5)
arnold2_detrended = detrend_fold_profiles(arnold2_decompacted)
## Make lower bound with less good params 
arnold3_decompacted = decompact_fold(arnold2, sediment_density=2000, tolerance=0.00000001, phi0=0.55, Lambda=2.5)
arnold3_detrended = detrend_fold_profiles(arnold2_decompacted)
# Plot the data. Note this code is dirty - sill_amp_from_time2_lower and the others all have the same name column, even though it contains differen things...
#plt.plot(arnold2_detrended['x_utm'], arnold2_detrended['decompacted_profile_detrended_km']*1000, label='Decompacted fold')
plt.plot(arnold2_detrended['x_utm'], arnold2_detrended['fold_profile_detrended'], label='Fold', color='g')
plt.fill_between(arnold2_detrended['x_utm'], arnold2_detrended['decompacted_profile_detrended_km']*1000, arnold3_detrended['decompacted_profile_detrended_km']*1000, color='b', alpha=0.2, label='Decompacted fold')
## Find how much 25m (thermal aureole) would compact by. Thickness = Thickness_0(1-phi_0)/(1-phi), then subtract this from the origional thickness
25 - (25*(1-0.4))/(1-0.15)
## Plot with the thermal aureole accounted for. 
plt.fill_between(arnold2_detrended['x_utm'], arnold2_detrended['decompacted_profile_detrended_km']*1000-7.3, arnold3_detrended['decompacted_profile_detrended_km']*1000-7.3, color='g', alpha=0.2, label='Decompacted fold and aureole')
plt.plot(sill_amp_from_time2['x_utm'], sill_amp_from_time2['sill_amp_meters'], label='Sill amplitude', color='r')
plt.fill_between(sill_amp_from_time2['x_utm'], sill_amp_from_time2_lower['sill_amp_meters'], sill_amp_from_time2_upper['sill_amp_meters'], color='r', alpha=0.2)
plt.ylabel("Sill thickness / Fold amplitude (meters)")
plt.xlabel("x utm")
plt.legend(loc=8)
plt.savefig("/home/mxh909/Desktop/magee_resolution_images/Resolution-decomp_fold2.pdf",bbox_inches='tight')
plt.show()

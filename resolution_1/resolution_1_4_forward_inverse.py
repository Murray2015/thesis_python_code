# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:01:31 2017

@author: murray
"""

## Import dependencies 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

## Read in wireline log data from .las files
log_data = pd.read_csv("Wireline/resolution-1_final.las", delim_whitespace=True, 
                       header=None, names=['DEPTH', 'BS', 'CALI', 'DTC', 'GR', 'GR_CORR', 
                       'RESD', 'RESS', 'SP'], na_values=-999.2500, skiprows=56)
                       
## Keep a copy incase of need to check unaltered params. 
log_data_original = log_data.copy()

## Fill log data with dummy values for water and sed. Create a dataframe of the filled values, 
## then concat to top of log_data.
# Water depth is looked up from paper logs
water_depth = 64.0 # meters
# Get the first recorded depth in the log (note shallow unconsolidate section rarely logged) 
first_depth_in_well = log_data['DEPTH'][0]
# Find the depth sampling interval 
depth_sample_interval = np.round(log_data['DEPTH'][1] - log_data['DEPTH'][0], decimals=4)
# Create a vector of depths to fill 
depths = np.arange(start=0,stop=first_depth_in_well, step=depth_sample_interval)
# Create a vector of bit sizes
BS = np.repeat(log_data['BS'][0],len(depths))
# Create a vector of NaNs for logs not used in analysis.
CALI= np.repeat(np.nan,len(depths))
GR= np.repeat(np.nan,len(depths))
GR_CORR= np.repeat(np.nan,len(depths))
RESD= np.repeat(np.nan,len(depths))
RESS= np.repeat(np.nan,len(depths))
SP= np.repeat(np.nan,len(depths))
# Create two vectors of velocity in the water and shalow sed, and concat.
DTC_water = np.repeat(1/(1500*3.28084/1e6),water_depth)
DTC_sed = np.repeat(1/(1800*3.28084/1e6),len(depths)-water_depth) 
DTC=np.insert(DTC_sed, 0, DTC_water)
# Make a temp dataframe of the newly generated data and NaNs, and then concat to make the full data.
temp_dataframe = pd.DataFrame({'DEPTH':depths, 'BS':BS, 'CALI':CALI, 'DTC':DTC, 'GR':GR, 'GR_CORR':GR_CORR, 'RESD':RESD, 'RESS':RESS, 'SP':SP})
log_data = pd.concat((temp_dataframe,log_data), ignore_index=True)

# Interpolate missing values in the logs (different tools turned on at different times) to enable the cumulative integration 
log_data.loc[0,'DT_s_per_m'] = (1.0/1500)
log_data['DT_s_per_m'] = log_data['DT_s_per_m'].interpolate()
log_data['DTC'] = log_data['DTC'].interpolate()

# Make a Vp log, to plot and also to later find the sill velocity from. 
log_data['VP_m_per_s'] = 1/(log_data['DTC'] * (1e-6) / 0.3048)

#########################################
############### Load files ##############
#########################################
## Read in horizons exported from kingdom. 
def read_horizon_files(file):
    assert type(file) == str, "File not a string"
    return pd.read_csv(file, sep='\s+', header=None, names=['x_utm','y_utm','line','trace','depth','amplitude'], engine='python')

sea_floor = read_horizon_files('/home/murray/Documents/thesis_python_code/resolution_1/resolution_depth_horizons_23-5-2017/sea_floor.dat')
top_forced_fold = read_horizon_files('/home/murray/Documents/thesis_python_code/resolution_1/resolution_depth_horizons_23-5-2017/top_forced_fold.dat')
southland = read_horizon_files('/home/murray/Documents/thesis_python_code/resolution_1/resolution_depth_horizons_23-5-2017/southland.dat')

top_sill_time = read_horizon_files('/home/murray/Documents/thesis_python_code/resolution_1/resolution_depth_horizons_23-5-2017/sill_time_grids/top_sill_time.dat')
base_sill_time = read_horizon_files('/home/murray/Documents/thesis_python_code/resolution_1/resolution_depth_horizons_23-5-2017/sill_time_grids/base_sill_time_2.dat')
top_sill_time = top_sill_time.rename(columns = {'depth':'time'})
base_sill_time = base_sill_time.rename(columns = {'depth':'time'})


def horizon_clip(horizon, sill):
    '''
    Function to extract a horizon above a sill. It DOES NOT
    subtract the background trend to find the actual amplitude. 
    The values 865 to 651 are where the forced fold starts and ends.

    '''
    before_sill = horizon.loc[(horizon['trace']>651) & (horizon['trace']<min(sill['trace']))]
    after_sill = horizon.loc[(horizon['trace']> max(sill['trace'])) & (horizon['trace'] < 865)]
    return before_sill.append([horizon[horizon['trace'].isin(sill['trace'])], after_sill])

sea_floor_clip = horizon_clip(sea_floor, base_sill_time).reset_index(drop=True)
top_forced_fold_clip = horizon_clip(top_forced_fold, base_sill_time).reset_index(drop=True)
southland_clip = horizon_clip(southland, base_sill_time).reset_index(drop=True)

top_sill_time= horizon_clip(top_sill_time, southland_clip).reset_index(drop=True)
base_sill_time= horizon_clip(base_sill_time, southland_clip).reset_index(drop=True)

plt.plot(sea_floor_clip['trace'], sea_floor_clip['depth'], label='seafloor')
plt.plot(southland_clip['trace'], southland_clip['depth'], label='southland')
plt.plot(top_forced_fold_clip['trace'], top_forced_fold_clip['depth'], label='forcedfold')
plt.plot(top_sill_time['trace'], top_sill_time['time']*1000, label='top sill')
plt.plot(base_sill_time['trace'], base_sill_time['time']*1000, label='base sill')
plt.gca().invert_yaxis()
plt.legend()
plt.show()
    
#######################################
######### Find Sill Thickness #########
#######################################
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

sill_amp_from_time = time_sill_amp_extraction_2d(top_sill_time, base_sill_time, sill_vel=6000)

##################################
########## Decompact #############
##################################
def decompact_fold(fold, z3=0, tolerance=0.000001, phi0=0.6, Lambda=2):
    '''
    This function decompacts a forced fold by repeatedly calling the function decompact
    '''
    z4_output = []   
    
    def decompact(z1, z2, z3, tolerance, phi0, Lambda):
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
        z4_output.append(decompact(fold['depth'].iloc[i]/1000, fold['depth'].max()/1000, z3, tolerance, phi0, Lambda))    
    fold['decompacted_profile_km'] = pd.Series(z4_output, index=fold.index)
    return fold 

fold_decompacted = decompact_fold(top_forced_fold_clip, tolerance=0.0001, phi0=0.6, Lambda=2)
southland_decompacted = decompact_fold(southland_clip, tolerance=0.0001, phi0=0.6, Lambda=2)

plt.plot(fold_decompacted['trace'], fold_decompacted['decompacted_profile_km'], label='fold')
plt.plot(southland_decompacted['trace'], southland_decompacted['decompacted_profile_km'], label='southland')
plt.plot(sill_amp_from_time['trace'], sill_amp_from_time['sill_amp_meters']/1000, label='sill thickness')
plt.legend()
plt.show()

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
southland_detrended = detrend_fold_profiles(southland_decompacted)

plt.plot(fold_detrended['trace'], fold_detrended['decompacted_profile_detrended_km'], label='decompacted fold')
plt.plot(southland_detrended['trace'], southland_detrended['decompacted_profile_detrended_km'], label='decompacted southland')
plt.plot(sill_amp_from_time['trace'], sill_amp_from_time['sill_amp_meters']/1000, label='sill thickness')
plt.legend()
plt.show()

##################################
######### Forward model ##########
##################################
greatest_sill_vel = 7000
least_sill_vel = 4500
greatest_phi0 = 0.7
least_phi0=0.25
greatest_lambda = 3.7
least_lambda = 1.4

## Make upper limit of fold profile 
greatest_fold_decompacted = decompact_fold(top_forced_fold_clip, tolerance=0.0001, phi0=greatest_phi0, Lambda=greatest_lambda)
greatest_southland_decompacted = decompact_fold(southland_clip, tolerance=0.0001, phi0=greatest_phi0, Lambda=greatest_lambda)
greatest_fold_detrended = detrend_fold_profiles(greatest_fold_decompacted)
greatest_southland_detrended = detrend_fold_profiles(greatest_southland_decompacted)

## Make lower limit of fold profile 
least_fold_decompacted = decompact_fold(top_forced_fold_clip, tolerance=0.0001, phi0=least_phi0, Lambda=least_lambda)
least_southland_decompacted = decompact_fold(southland_clip, tolerance=0.0001, phi0=least_phi0, Lambda=least_lambda)
least_fold_detrended = detrend_fold_profiles(least_fold_decompacted)
least_southland_detrended = detrend_fold_profiles(least_southland_decompacted)

## Make upper limit of sill thickness 
greatest_sill_amp_from_time = time_sill_amp_extraction_2d(top_sill_time, base_sill_time, sill_vel=greatest_sill_vel)

## Make lower limit of sill thickness 
least_sill_amp_from_time = time_sill_amp_extraction_2d(top_sill_time, base_sill_time, sill_vel=least_sill_vel)

## Plot 
plt.fill_between(greatest_fold_detrended['trace'], least_fold_detrended['decompacted_profile_detrended_km']*1000, greatest_fold_detrended['decompacted_profile_detrended_km']*1000, color='b', alpha=0.2, label='Decompacted fold')
plt.fill_between(greatest_southland_detrended['trace'], least_southland_detrended['decompacted_profile_detrended_km']*1000, greatest_southland_detrended['decompacted_profile_detrended_km']*1000, color='g', alpha=0.2, label='Decompacted Southland')
plt.fill_between(greatest_sill_amp_from_time['trace'], least_sill_amp_from_time['sill_amp_meters'], greatest_sill_amp_from_time['sill_amp_meters'], color='r', alpha=0.2, label='Sill thickness')
plt.ylabel("Sill thickness / Fold amplitude (meters)")
plt.xlabel("Trace")
plt.legend(loc=8)
plt.savefig("/home/murray/Documents/thesis_python_code/Resolution_forward_model.pdf",bbox_inches='tight')
plt.show()



##################################
######### Inverse model ##########
##################################

## Set vars 
greatest_sill_vel = 7000
least_sill_vel = 4500
greatest_phi0 =0.7
least_phi0=0.25
phi0_decimation = 0.05
greatest_lambda = 3.7
least_lambda = 1.4
lambda_decimation = 0.1
phi0_vec = np.arange(start=least_phi0, stop=greatest_phi0, step=phi0_decimation)
lambda_vec = np.arange(start=least_lambda, stop=greatest_lambda, step=lambda_decimation)
grid_search = np.zeros((len(phi0_vec), len(lambda_vec)))


## Misfit function 
def misfit(sill, forced_fold):
    ''' 
    Function which returns the misfit between a sill curve and a 
    forced fold curve.
    '''
    total_misfit = 0 
    for i in range(len(sill)):
        if sill['trace'][i] in forced_fold['trace'].values:
            total_misfit += (sill['sill_amp_meters'][i]/1000.0) - forced_fold['decompacted_profile_detrended_km'].loc[forced_fold['trace']==sill['trace'][i]].values
    return total_misfit
    
#misfit(sill_amp_from_time, fold_detrended)

## Grid search
## Outer loop - phi0
for i_loc, i in enumerate(phi0_vec):
    ## Inner loop - lambda
    for j_loc, j in enumerate(lambda_vec):
        ## Note, could do a third loop for sill vel
        sill_amp_gs = time_sill_amp_extraction_2d(top_sill_time, base_sill_time, sill_vel=5500)
        fold_gs = detrend_fold_profiles(decompact_fold(top_forced_fold_clip, tolerance=0.0001, phi0=i, Lambda=j))
        grid_search[i_loc,j_loc] = misfit(sill_amp_gs, fold_gs) ## fill this with the misfit function. 
        print('i = ', i_loc, ', j = ', j_loc, ', total = ', j_loc+(len(lambda_vec)*i_loc), 'out of ', len(phi0_vec)*len(lambda_vec))



## Plot the grid searc
g2 = abs(grid_search)
max_loc = np.where(g2 == g2.min())
opt_phi0 = phi0_vec[::-1][max_loc[0]]
opt_lambda = lambda_vec[max_loc[1]]
plt.figure()
plt.imshow(g2, aspect='equal', cmap='jet_r', extent=[least_lambda, greatest_lambda, least_phi0, greatest_phi0])
#plt.scatter(y=[opt_phi0], x=[opt_lambda], c='r', s=40)
plt.colorbar(cmap='jet_r')
plt.ylabel(r'$\Phi_0$')
plt.xlabel(r'$\lambda$')
plt.savefig("/home/murray/Documents/thesis_python_code/Resolution_inverse_grid.pdf",bbox_inches='tight')
plt.show()

## Next to do - change code so the white padding at the base of the image goes. 
# add contours
# replot with more squares 
# find why lambda axes doesn't go to lambda max (def a bug somewhere)

## Plot fold profile with minimum misfit 
fold_decompacted = decompact_fold(top_forced_fold_clip, tolerance=0.0001, phi0=opt_phi0, Lambda=opt_lambda)
southland_decompacted = decompact_fold(southland_clip, tolerance=0.0001, phi0=opt_phi0, Lambda=opt_lambda)

fold_detrended = detrend_fold_profiles(fold_decompacted)
southland_detrended = detrend_fold_profiles(southland_decompacted)

plt.plot(fold_detrended['trace'], fold_detrended['decompacted_profile_detrended_km']*1000, label='Decompacted fold')
plt.plot(southland_detrended['trace'], southland_detrended['decompacted_profile_detrended_km']*1000, label='Decompacted Southland')
plt.fill_between(greatest_sill_amp_from_time['trace'], least_sill_amp_from_time['sill_amp_meters'], greatest_sill_amp_from_time['sill_amp_meters'], color='r', alpha=0.2, label='Sill thickness')
plt.ylabel("Sill thickness / Fold amplitude (meters)")
plt.xlabel("Trace")
plt.legend(loc=8)
plt.savefig("/home/murray/Documents/thesis_python_code/Resolution_inverse_model.pdf",bbox_inches='tight')
plt.show()


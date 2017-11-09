# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:01:31 2017

@author: Murray Hoggett, murrayhoggett@gmail.com. 
"""

## Import dependencies 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np


#########################################
############### Load files ##############
#########################################
## Read in seismic horizons exported from SMT kingdom interpretation software. 
def read_horizon_files(file_name):
    assert type(file_name) == str, "File not a string"
    return pd.read_csv(file_name, sep='\s+', header=None, names=['x_utm','y_utm','line','trace','depth','amplitude'], engine='python')

sea_floor = read_horizon_files('/home/murray/Documents/thesis_python_code/resolution_1/resolution_depth_horizons_23-5-2017/sea_floor.dat')
southland = read_horizon_files('/home/murray/Documents/thesis_python_code/resolution_1/resolution_depth_horizons_23-5-2017/southland.dat')
top_sill_time = read_horizon_files('/home/murray/Documents/thesis_python_code/resolution_1/resolution_depth_horizons_23-5-2017/sill_time_grids/top_sill_time.dat')
base_sill_time = read_horizon_files('/home/murray/Documents/thesis_python_code/resolution_1/resolution_depth_horizons_23-5-2017/sill_time_grids/base_sill_time_2.dat')
# Rename the columns for the sills, as these are exported from kingdom in time, not depth.
top_sill_time = top_sill_time.rename(columns = {'depth':'time'})
base_sill_time = base_sill_time.rename(columns = {'depth':'time'})


def horizon_clip(horizon, sill):
    '''
    Function to extract a horizon above a sill (the fold), matching traces for later 
    computation. It DOES NOT subtract the background trend to find the actual fold 
    amplitude. The values 865 to 651 are where the forced fold starts and ends.

    '''
    before_sill = horizon.loc[(horizon['trace']>651) & (horizon['trace']<min(sill['trace']))]
    after_sill = horizon.loc[(horizon['trace']> max(sill['trace'])) & (horizon['trace'] < 865)]
    return before_sill.append([horizon[horizon['trace'].isin(sill['trace'])], after_sill])

sea_floor_clip = horizon_clip(sea_floor, base_sill_time).reset_index(drop=True)
southland_clip = horizon_clip(southland, base_sill_time).reset_index(drop=True)
top_sill_time= horizon_clip(top_sill_time, southland_clip).reset_index(drop=True)
base_sill_time= horizon_clip(base_sill_time, southland_clip).reset_index(drop=True)

    
#######################################
######### Find Sill Thickness #########
#######################################
def time_sill_amp_extraction_2d(top,base,sill_vel):
    '''
    Function to test for corresponding points in sill top and base. If found to 
    correspond, the function subtracts the top from the base to give the sill 
    amplitude in milliseconds. 
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

sill_amp_from_time = time_sill_amp_extraction_2d(top_sill_time, base_sill_time, sill_vel=6500)

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
        of z2 is called z4. Note z1,z2 and z3 must be given in kilometers. 
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

southland_decompacted = decompact_fold(southland_clip, tolerance=0.0001, phi0=0.3, Lambda=2)
plt.plot(southland_decompacted['trace'], southland_decompacted['decompacted_profile_km'], label='southland')
plt.plot(sill_amp_from_time['trace'], sill_amp_from_time['sill_amp_meters']/1000, label='sill thickness')
plt.legend()
plt.show()

##################################
############ Detrend #############
##################################
def detrend_fold_profiles(fold):
    '''
    This function detrends the fold profile to remove the effect of being on a shallow slope.
    We fit a linear trend between the first and last points in the file, and then
    subtract that trend from the points in the file. 
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
    m2 = (last_point2[1] - first_point2[1])/(last_point2[0]-first_point2[0])
    b2 = first_point2[1] - m2*(first_point2[0])
    fold2['fold_profile_detrended'] = fold2['depth']
    fold2['fold_profile_detrended'] -= (fold2['x_utm']*m2 + b2)
    fold2['fold_profile_detrended'] = abs(fold2['fold_profile_detrended']) 
    return fold2

southland_detrended = detrend_fold_profiles(southland_decompacted)
plt.plot(southland_detrended['trace'], southland_detrended['decompacted_profile_detrended_km'], label='decompacted southland')
plt.plot(sill_amp_from_time['trace'], sill_amp_from_time['sill_amp_meters']/1000, label='sill thickness')
plt.legend()
plt.show()


##################################
######### Forward model ##########
##################################
## First we make forward models using all sensible parameter ranges. We use the 
## Southland horizon just below the forced fold, as it shows no signs of erosion. 
greatest_sill_vel = 7000
least_sill_vel = 4500
greatest_phi0 = 0.7
least_phi0=0.20
greatest_lambda = 3.7
least_lambda = 1.4

## Make upper limit of fold profile 
greatest_southland_decompacted = decompact_fold(southland_clip, tolerance=0.0001, phi0=greatest_phi0, Lambda=greatest_lambda)
greatest_southland_detrended = detrend_fold_profiles(greatest_southland_decompacted)
## Make lower limit of fold profile 
least_southland_decompacted = decompact_fold(southland_clip, tolerance=0.0001, phi0=least_phi0, Lambda=least_lambda)
least_southland_detrended = detrend_fold_profiles(least_southland_decompacted)
## Make upper limit of sill thickness 
greatest_sill_amp_from_time = time_sill_amp_extraction_2d(top_sill_time, base_sill_time, sill_vel=greatest_sill_vel)
## Make lower limit of sill thickness 
least_sill_amp_from_time = time_sill_amp_extraction_2d(top_sill_time, base_sill_time, sill_vel=least_sill_vel)
## Plot 
plt.fill_between(greatest_southland_detrended['trace'], least_southland_detrended['decompacted_profile_detrended_km']*1000, greatest_southland_detrended['decompacted_profile_detrended_km']*1000, color='g', alpha=0.2, label='Decompacted Southland')
plt.fill_between(greatest_sill_amp_from_time['trace'], least_sill_amp_from_time['sill_amp_meters'], greatest_sill_amp_from_time['sill_amp_meters'], color='r', alpha=0.2, label='Sill thickness')
plt.ylabel("Sill thickness / Fold amplitude (meters)")
plt.xlabel("Trace")
plt.legend(loc=8)
#plt.savefig("/home/murray/Documents/thesis_python_code/Resolution_forward_model.pdf",bbox_inches='tight')
plt.show()



##################################
######### Inverse model ##########
##################################
## Secondly, we solve the inverse problem to find the best fitting decompaction
## parameters to fit the sill. We use a grid search 

## Set vars 
greatest_sill_vel = 7000
least_sill_vel = 4500
greatest_phi0 =0.7
least_phi0=0.20
phi0_decimation = 0.01
greatest_lambda = 3.7
least_lambda = 1.4
lambda_decimation = 0.05
phi0_vec = np.arange(start=least_phi0, stop=greatest_phi0, step=phi0_decimation)
lambda_vec = np.arange(start=least_lambda, stop=greatest_lambda, step=lambda_decimation)
grid_search = np.zeros((len(phi0_vec), len(lambda_vec)))

## Misfit function 
def RMS_misfit(sill, forced_fold):
    ''' 
    Function which returns the root mean squared misfit between a sill curve and a 
    forced fold curve.
    '''
    total_misfit = 0 
    counter = 0
    for i in range(len(sill)):
        if sill['trace'][i] in forced_fold['trace'].values:
            counter += 1
            total_misfit += np.power((sill['sill_amp_meters'][i]/1000.0) - forced_fold['decompacted_profile_detrended_km'].loc[forced_fold['trace']==sill['trace'][i]].values, 2)
    rms_misfit = np.power((total_misfit / counter), 0.5)
    return rms_misfit

## Grid search
## Outer loop - phi0
sill_amp_gs = time_sill_amp_extraction_2d(top_sill_time, base_sill_time, sill_vel=6000)
for i_loc, i in enumerate(phi0_vec):
    ## Inner loop - lambda
    for j_loc, j in enumerate(lambda_vec):
        ## Note, could do a third loop for sill vel
        fold_gs = detrend_fold_profiles(decompact_fold(southland_clip, tolerance=0.0001, phi0=i, Lambda=j))
        grid_search[len(phi0_vec)-(i_loc+1),j_loc] = RMS_misfit(sill_amp_gs, fold_gs) ## fill this with the misfit function. 
        print('i = ', i_loc, ', j = ', j_loc, ', total = ', j_loc+(len(lambda_vec)*i_loc), 'out of ', len(phi0_vec)*len(lambda_vec))

## Find the optimum values 
max_loc = np.where(grid_search == grid_search.min())
opt_phi0 = phi0_vec[::-1][max_loc[0]]
opt_lambda = lambda_vec[max_loc[1]]

## Plot the grid searc
plt.figure(figsize=(5,4))
plt.contour(grid_search, 20, linewidths=0.5, colors='k', aspect='auto', extent=[least_lambda, greatest_lambda, greatest_phi0, least_phi0])
plt.contourf(grid_search, 20, aspect='auto', cmap='jet_r', extent=[least_lambda, greatest_lambda, greatest_phi0, least_phi0])
cbar = plt.colorbar(cmap='jet_r')
cbar.ax.set_ylabel('RMS Misfit')
plt.scatter(opt_lambda, opt_phi0, marker='o', s=100, c='k')
plt.ylabel(r'$\Phi_0$')
plt.xlabel(r'$\lambda$')
#plt.savefig("/home/murray/Documents/thesis_python_code/Resolution_inverse_grid.pdf",bbox_inches='tight')
plt.show()

## Plot fold profile with minimum misfit 
southland_decompacted = decompact_fold(southland_clip, tolerance=0.0001, phi0=opt_phi0, Lambda=opt_lambda)
southland_detrended = detrend_fold_profiles(southland_decompacted)
plt.plot(southland_detrended['trace'], southland_detrended['decompacted_profile_detrended_km']*1000, label='Decompacted Southland')
plt.fill_between(greatest_sill_amp_from_time['trace'], least_sill_amp_from_time['sill_amp_meters'], greatest_sill_amp_from_time['sill_amp_meters'], color='r', alpha=0.2, label='Sill thickness')
plt.ylabel("Sill thickness / Fold amplitude (meters)")
plt.xlabel("Trace")
plt.legend(loc=8)
#plt.savefig("/home/murray/Documents/thesis_python_code/Resolution_optimum_inverse_model.pdf",bbox_inches='tight')
plt.show()


##################################
### Matrix property inversion ####
##################################
### Finally, make a theoretical porosity curve from the above inversion results, 
### fit to the porosity curve with 1D exhaustive enumeration, and plot the misfit. 

## Read in wireline log data from .las files
log_data = pd.read_csv("Wireline/resolution-1_final.las", delim_whitespace=True, 
                       header=None, names=['DEPTH', 'BS', 'CALI', 'DTC', 'GR', 'GR_CORR', 
                       'RESD', 'RESS', 'SP'], na_values=-999.2500, skiprows=56)
                       
## Fill log data with dummy values for water and sed. Create a dataframe of the filled values, 
## then concat to top of log_data.
# Water depth is found from well comp. report
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


# Get array of depths
log_depths = log_data['DEPTH'].values/1000.00
# Apply formula to find the poro at those depths 
theoretical_poro = opt_phi0 * np.exp(-(log_depths) / opt_lambda)


# Define RHG function to change sonic to poro 
def RHG(well_log, matrix_vel):
    '''Function to calculate Raymer-Hunt-Gardner porosity from Sonic log'''
    return (5.0/8)*((log_data['DTC'] - matrix_vel)/(log_data['DTC']))
    
## Define misfit function for RHG vs. theoretical poro
def RHG_misfit(log_poro, theoretical_poro, range_of_interest=(2145, 12100)):
    '''Function to calculate the misfit between a theoretical porosity log and a calculated
    porosity log ''' 
    misfit_vec = np.power((log_poro[range_of_interest[0]:range_of_interest[1]] - theoretical_poro[range_of_interest[0]:range_of_interest[1]]), 2)
    return np.sqrt(np.divide(np.sum(misfit_vec), len(misfit_vec)))


## Define range of matrix velocities 
matrix_vels = np.arange(start=40, stop=90, step=1)

## Define vector of zeroes for misfit results 
misfits = np.zeros(len(matrix_vels))

## Loop over matrix vels to find the misfit for the matrix velocities 
for i_loc, i in enumerate(matrix_vels):
    misfits[i_loc] = RHG_misfit(RHG(log_data, i), theoretical_poro, range_of_interest=(0, 12100))

best_matrix_vel = matrix_vels[np.argmin(misfits)]
plt.plot(matrix_vels, abs(misfits))
plt.axvline(x=best_matrix_vel, c='r')
plt.text(best_matrix_vel+2.5, 0.12, r'$Minimum = %s \mu s / ft $' % (best_matrix_vel), color='r')
plt.ylabel("RMSE misfit")
plt.xlabel(r"$Matrix \ velocity \ (\mu s / ft)$")
plt.savefig("/home/murray/Documents/thesis_python_code/Resolution_inverse_matrix_misfit.pdf",bbox_inches='tight')
plt.show()

## Plot test curves of theoretical poro and cal poro, to check everything
rhg = RHG(log_data, best_matrix_vel)
plt.figure(figsize=(4,6))
plt.plot(theoretical_poro, log_depths, label='Theoretical porosity')
plt.plot(rhg, log_depths, label='Calculated porosity')
plt.plot(rhg[2145:12100], log_depths[2145:12100], label='Fitted region')
plt.legend()
plt.xlabel(r"$\phi$")
plt.ylabel('Depth, km')
plt.gca().invert_yaxis()
plt.xlim(0,0.6)
plt.savefig("/home/murray/Documents/thesis_python_code/Resolution_inverse_log.pdf",bbox_inches='tight')
plt.show()

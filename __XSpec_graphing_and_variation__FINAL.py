# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 11:12:30 2024

@author: deand
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import numpy as np

from pathlib import Path
from scipy import integrate

import astropy.io.fits
from tqdm import tqdm


###############################################################################

"""
NOTE BEFORE RUNNING THE CODE:::
    
    make sure the Xrad2 program is on the same directory as the python code
    this code is separated into cells
"""




#%% CONTROL PARAMETERS #########################################################

### line 1 - observer parameters ##############################################
viewing_angle_cont = 60.
distance_cont = 1000.
zoom_out_fct_cont = 1.

inp_lin1_cont = "{} {} {}".format(viewing_angle_cont, 
                                  distance_cont, 
                                  zoom_out_fct_cont)

### line 2 - disk params ######################################################
angular_momentum_cont = 0.5
R_in_cont = .1
R_out_cont = 19.
emissivity_idx_cont = 2.5

inp_lin2_cont = "{} {} {} {}".format(angular_momentum_cont, 
                                     R_in_cont, 
                                     R_out_cont, 
                                     emissivity_idx_cont)

### line 3 - absorption region params #########################################
seed = -13 #does not change
num_abs_cont = 1
abs_rad_cont = 0.2
center_x_coord_cont = 0.
center_y_coord_cont = 0.
std_dev_cont = 7.

inp_lin3_cont = "{} {} {} {} {} {}".format(seed, 
                                           num_abs_cont,
                                           abs_rad_cont,
                                           center_x_coord_cont,
                                           center_y_coord_cont,
                                           std_dev_cont)

### line 4 - absorption coefficient params ####################################
line_rest_nrg_cont = 6.4
abs_int_cont = -2
central_nrg_cont = 7.4
abs_width_band_cont = 0.5

inp_lin4_cont = "{} {} {} {}".format(line_rest_nrg_cont,
                                     abs_int_cont,
                                     central_nrg_cont,
                                     abs_width_band_cont)

### line 5 - image param ### DOES NOT CHANGE ##################################
disk_res = 300
line_prof_res = 100

inp_lin5_cont = "300 100 'xrad2.eps/vcps'"


###############################################################################



#%% defining the function and procedure one by one, for clarity #################
## running the file, should give an input of xrad2.lin ########################

"""
this is a procedure to run the xrad2 file for each lines.
input: line1, ..., line5 = line by line variation of xrad2.exe code input
output: data_energy = energy of data (in keV)
        data_flux_(un)abs = (un)absorbed flux generated (in arbitrary unit, max of unabsorbed = 1)
"""
def xrad2run (line1 = inp_lin1_cont, 
              line2 = inp_lin2_cont, 
              line3 = inp_lin3_cont,
              line4 = inp_lin4_cont,
              line5 = inp_lin5_cont): #assign control variables as defaults

    inp_file = []
    inp_file.append(line1)
    inp_file.append(line2)
    inp_file.append(line3)
    inp_file.append(line4)
    inp_file.append(line5)
    
    np.savetxt('xrad2.inp', np.array(inp_file), fmt='%s')
        
    os.system(f"xrad2.exe") #running external program

    
    #reading the .lin file from xrad2.exe output
    colspecs = [(3, 11), (14, 21), (23, 31)]  # define column widths
    df = pd.read_fwf(r'xrad2.lin', 
                     colspecs=colspecs, header=None,
                     names=['energy', 'flux_unabsorbed', 'flux_absorbed'], skiprows=7)
            #energy is in keV, flux is in arbitrary units (normalized)
        
        
    data_energy = df.energy
    
    data_flux_unabs = df.flux_unabsorbed
    data_flux_abs = df.flux_absorbed

    return data_energy, data_flux_unabs, data_flux_abs

## lines parameters, from the first 8 rows of xrad2.lin #######################
"""
there are 6 parameters from xrad2.lin files that are directly stated. 
"""
def xrad2lin_param_read ():
    colspecs = [(30, 40)]
    df = pd.read_fwf(r'xrad2.lin', colspecs=colspecs, header=None, nrows=7)
    df = df[0]
    
    #change all (except percentage) into float (0, 1, 2 index) or integer (3, 4)
    df[0:3] = np.float64(df[0:3])
    df[3] = np.int0(df[3])
    df[4] = np.int0(df[4])
    
    #removing the percentage into float
    percent = df[5]
    percent_num = np.float64(percent[:-1])
    df[5] = percent_num
    return list(df)

## FWHM detection #############################################################
"""
detection of FWHM, unabs and abs
2 method, line, or area. Area is done if there are there are fluctuation of 
above the half maximum (see when r_in = 1, r_out=100) area is calculated by 
performing integration for flux-half. if flux-half <0, then the value is set to 0.

in addition, graphical representation is also added, but for now, it's line only.

input: data_energy, data_flux_(un)abs = xrad2.lin file outputs, generated from above fcts.
output: fwhm value for absorbed and unabsorbed.
"""

def fwhm_calc (data_energy, data_flux_unabs, data_flux_abs, 
               graph = 'False',
               AreaMode = 'False'):
    
    # unabsorbed FWHM determination -> should be measured by the area of the graph, but i'll just measure the width instead.
    more_half_index = np.array(np.where(data_flux_unabs > 0.5)).flatten() #max = 1 is guaranteed for unabsorbed
    fwhm_unabs_left = data_energy[more_half_index[0]]
    fwhm_unabs_right = data_energy[more_half_index[-1]]

    fwhm_unabs = np.float32(fwhm_unabs_right - fwhm_unabs_left) #convert to float32 to reduce float numbers

    # absorbed FWHM determination
    half_point_abs = 0.5*(np.max(data_flux_abs))
    more_half_index_abs = np.array(np.where(data_flux_abs > half_point_abs)).flatten() 
    fwhm_abs_left = data_energy[more_half_index_abs[0]]
    fwhm_abs_right = data_energy[more_half_index_abs[-1]]
    
    fwhm_abs = np.float32(fwhm_abs_right - fwhm_abs_left) #convert to float32 to reduce float numbers


    if graph == 'True':
        # plotting the spectrum
        ## flux plotting
        plt.plot(data_energy, data_flux_unabs, label='unabsorbed')
        plt.plot(data_energy, data_flux_abs, label='absorbed', linestyle='--')

        ## fwhm plotting // unabsorbed
        fwhm_unabs_line = np.linspace(fwhm_unabs_left, fwhm_unabs_right)
        fwhm_unabs_plot = np.linspace(0.5, 0.5)

        plt.plot(fwhm_unabs_line, fwhm_unabs_plot, label='unabsorbed FWHM')

        ## fwhm ploting // absorbec
        fwhm_abs_line = np.linspace(fwhm_abs_left, fwhm_abs_right)
        fwhm_abs_plot = np.linspace(half_point_abs, half_point_abs)

        plt.plot(fwhm_abs_line, fwhm_abs_plot, label='absorbed FWHM')

        plt.xlabel('energy (keV)')
        plt.ylabel('flux (normalized arbitrary units)')
        plt.legend()
        plt.show()
        
    if AreaMode == 'True':
        flux_calc_unabs = data_flux_unabs-0.5
        flux_calc_unabs[flux_calc_unabs<0.]=0.
        
        flux_calc_abs = data_flux_abs-half_point_abs
        flux_calc_abs[flux_calc_abs<0.]=0.
        
        fwhm_unabs = integrate.simpson(flux_calc_unabs, data_energy)
        fwhm_abs = integrate.simpson(flux_calc_abs, data_energy)
    
    return fwhm_unabs, fwhm_abs

## peak value and location detection ##########################################
"""
detection of peak value and location. main peak value is guaranteed to be 1.0.
precision is admittedly... minimal since the energy position is determined with 
integers, resulting in a discreet progression depending on the energy value.

in addition, if secondary peaks is dubious or undetected, a nan value is set.


input: data_energy, data_flux_(un)abs = xrad2.lin file outputs, generated from above fcts.
output: peak values and position for main and secondary, for absorbed and unabsorbed

ngl, this fct gives 8 outputs... which is too many imho
should make it an array just to be clear


"""
def peak_detection_calc (data_energy, data_flux_unabs, data_flux_abs):
    # main peak location and detection
    main_peak_unabs = np.max(data_flux_unabs)
    mp_loc_unabs = int(np.array(np.where(data_flux_unabs == main_peak_unabs)))

    main_peak_abs = np.max(data_flux_abs)
    mp_loc_abs = int(np.array(np.where(data_flux_abs == main_peak_abs)))

    # secondary peak detection (it's assumed that absorbed and secondary has the same peak position)
        # consider sample graph at the cobtrol, change to 500 absorber
        # "peak" located nearer to the peak has a... dubious quality


    #for unabsorbed lines
    for i in range(37, mp_loc_abs):       
        """
        list of dirty (simulation plot too variative, secondary peak is... guesseed manually)
                also listed the change of starting range
       
        - emissivity    = 4, 10     -> change to 45
        - emissivity    = 100       -> change to 47
        - inner/outer   = 0.1/100   -> change to 43
        - inner/outer   = 1/100     -> change to 43
        - angle         = 90        -> change to 43
        - angular mom.  = 1.00      -> change to 37
        """
        if ((data_flux_unabs[i] > data_flux_unabs[i-1]) & (data_flux_unabs[i] > data_flux_unabs[i+1])): #edit this to be the maximum, but with caution
            secondary_peak_index_unabs = i
            # print(i)
            break
        else:
            secondary_peak_index_unabs = []
            
            
    main_peak_flux_unabs = data_flux_unabs[mp_loc_unabs] #this should give 1 with all data
    main_peak_ener_unabs = data_energy[mp_loc_unabs]
        
    if secondary_peak_index_unabs:            #if secondary peak is detected
        secondary_peak_flux_unabs = data_flux_unabs[secondary_peak_index_unabs]
        secondary_peak_ener_unabs = data_energy[secondary_peak_index_unabs]


    else:   #failsafe if no second peak is detected, 
        secondary_peak_flux_unabs = np.nan
        secondary_peak_ener_unabs = np.nan
       
        
        
    
    #for absorbed lines
    for i in range(37, mp_loc_unabs):       

        if ((data_flux_abs[i] > data_flux_abs[i-1]) & (data_flux_abs[i] > data_flux_abs[i+1])): #edit this to be the maximum, but with caution
            secondary_peak_index_abs = i
            # print(i)
            break
        else:
            secondary_peak_index_abs = []
            
            
    main_peak_flux_abs = data_flux_abs[mp_loc_abs] #this should give 1 with all data
    main_peak_ener_abs = data_energy[mp_loc_abs]
        
    if secondary_peak_index_abs:            #if secondary peak is detected
        secondary_peak_flux_abs = data_flux_abs[secondary_peak_index_abs]
        secondary_peak_ener_abs = data_energy[secondary_peak_index_abs]


    else:   #failsafe if no second peak is detected, 
        secondary_peak_flux_abs = np.nan
        secondary_peak_ener_abs = np.nan


    #put the output into an array, so that it'll be neat
    output_arr = []
    output_arr.append(main_peak_flux_unabs)
    output_arr.append(main_peak_ener_unabs)
    output_arr.append(secondary_peak_flux_unabs)
    output_arr.append(secondary_peak_ener_unabs)    
    
    output_arr.append(main_peak_flux_abs)
    output_arr.append(main_peak_ener_abs)
    output_arr.append(secondary_peak_flux_abs)
    output_arr.append(secondary_peak_ener_abs)

    return output_arr




## putting it all together ####################################################
"""
to put it all together

input: line variations
output: all of the parameters, presented as arrays with following description and structure

xrad2lin_params
[0] unabsorbed line flux
[1] absorbed line flux
[2] flux ratio
[3] # of emitted rays
[4] # of absorbed rays
[5] coverage ratio (in percents)

fwhm_arr
[0] unabsorbed fwhm (line calculation)       
[1] unabsorbed fwhm (area calculation)
[2] absorbed fwhm (line calculation)         
[3] absorbed fwhm (line calculation)


peak_params
[0] main peak flux (unabsorbed)              
[1] main peak energy (unabsorbed)
[2] secondary peak flux (unabsorbed)         
[3] secondary peak energy (unabsorbed)

[4] main peak flux (absorbed)                
[5] main peak energy (absorbed)
[6] secondary peak flux (absorbed)           
[7[ secondary peak energy (absorbed)  
"""
def line_by_line_var (line1 = inp_lin1_cont, 
                      line2 = inp_lin2_cont, 
                      line3 = inp_lin3_cont,
                      line4 = inp_lin4_cont,
                      line5 = inp_lin5_cont): #assign control variables as defaults

    #get the spectra graph
    data_energy, data_flux_unabs, data_flux_abs = xrad2run(line1, line2, line3, line4, line5)
    
    #get the spectra parameters
    xrad2lin_params = xrad2lin_param_read()
    
    #get the FWHM, line and area, make it into array where 1st columns are line and 2nd are area
    fwhm_line_unabs, fwhm_line_abs = fwhm_calc(data_energy, data_flux_unabs, data_flux_abs)
    fwhm_area_unabs, fwhm_area_abs = fwhm_calc(data_energy, data_flux_unabs, data_flux_abs, AreaMode = 'True')
    
    fwhm_arr = []
    fwhm_arr.append(fwhm_line_unabs)
    fwhm_arr.append(fwhm_area_unabs)
    
    fwhm_arr.append(fwhm_line_abs)
    fwhm_arr.append(fwhm_area_abs)

    
    #get the peak parameters
    peak_params = peak_detection_calc(data_energy, data_flux_unabs, data_flux_abs)
    
    return xrad2lin_params, fwhm_arr, peak_params
############################################################################### 



## plotting function ##########################################################
"""
THIS STILL NEED WORKING, TANYA SAYYED LGI ANJIR SOAL INIIIIIIII

and to be plotted, all!
    focus on the fwhm_arr and the peak params

absorbed + unabsorbed join into one

update, plot some difference
"""
def plotting (inp_vars, xrad2lin_params, fwhm_arr, peak_params, var_name='varname forgor', unit=''):

#FWHM Parameters
    FWHM_plot = plt.figure(figsize=(12, 4))                        #figsize = 8,6 for FWHM ()
    FWHM_plot.suptitle('FWHM variation with {}'.format(var_name))
    spec = mpl.gridspec.GridSpec(ncols = 2, nrows=1)
    spec.update(wspace = 0.25, hspace = 0.25)
    
    # line
    fwhm_line = FWHM_plot.add_subplot(spec[0,0])
    fwhm_line.plot(inp_vars, fwhm_arr[0], label='unabsorbed')
    fwhm_line.plot(inp_vars, fwhm_arr[2], color='red', linestyle=(0, (2, 4)), label='absorbed')
    fwhm_line.legend()
    fwhm_line.set_title('FWHM with line calculation')
    fwhm_line.set_ylabel('FWHM (keV)')
    fwhm_line.set_xlabel('{} ({})'.format(var_name, unit))
    
    # area
    fwhm_area = FWHM_plot.add_subplot(spec[0,1])
    fwhm_area.plot(inp_vars, fwhm_arr[1], label='unabsorbed')
    fwhm_area.plot(inp_vars, fwhm_arr[3], color='red', linestyle=(0, (2, 4)), label='absorbed')
    fwhm_area.legend()
    fwhm_area.set_title('FWHM with area calculation')
    fwhm_area.set_ylabel('FWHM (keV * normalized flux)')
    fwhm_area.set_xlabel('{} ({})'.format(var_name, unit))
    
    # subplot viewing
    FWHM_plot
    
    
#peak Parameters
    peak_plot = plt.figure(figsize=(12, 12))                        
    peak_plot.suptitle('Peak variation with {}'.format(var_name))
    spec = mpl.gridspec.GridSpec(ncols = 2, nrows=3)
    spec.update(wspace = 0.25, hspace = 0.25, top = 0.93)
    
    # flux
    ## main
    flux_main = peak_plot.add_subplot(spec[0,0]) #flux (main peak)
    flux_main.plot(inp_vars, peak_params[0], label='unabsorbed')
    flux_main.plot(inp_vars, peak_params[4], color='red', linestyle=(0, (2, 4)), label='absorbed')
    flux_main.legend()
    flux_main.set_title('Main flux variation')
    flux_main.set_ylabel('Main flux (normalized flux)')
    flux_main.set_xlabel('{} ({})'.format(var_name, unit))
    
    ## sec
    flux_sec = peak_plot.add_subplot(spec[1,0]) #flux (main peak)
    flux_sec.plot(inp_vars, peak_params[2], label='unabsorbed')
    flux_sec.plot(inp_vars, peak_params[6], color='red', linestyle=(0, (2, 4)), label='absorbed')
    flux_sec.legend()
    flux_sec.set_title('Secondary flux variation')
    flux_sec.set_ylabel('Secondary flux (normalized flux)')
    flux_sec.set_xlabel('{} ({})'.format(var_name, unit))
    
    ## difference
    flux_diff_unabs = peak_params[0]-peak_params[2]
    flux_diff_abs = peak_params[4]-peak_params[6]
    
    flux_diff = peak_plot.add_subplot(spec[2,0]) #flux diff
    flux_diff.plot(inp_vars, flux_diff_unabs, label='unabsorbed')
    flux_diff.plot(inp_vars, flux_diff_abs, color='red', linestyle=(0, (2, 4)), label='absorbed')
    flux_diff.legend()
    flux_diff.set_title('Flux difference variation')
    flux_diff.set_ylabel('Flux difference (normalized flux)')
    flux_diff.set_xlabel('{} ({})'.format(var_name, unit))
    
    # energy
    ## main
    nrg_main = peak_plot.add_subplot(spec[0,1]) #energy (main peak)
    nrg_main.plot(inp_vars, peak_params[1], label='unabsorbed')
    nrg_main.plot(inp_vars, peak_params[5], color='red', linestyle=(0, (2, 4)), label='absorbed')
    nrg_main.legend()
    nrg_main.set_title('Main energy variation')
    nrg_main.set_ylabel('Main energy (keV)')
    nrg_main.set_xlabel('{} ({})'.format(var_name, unit))
    
    ## secondary
    nrg_sec = peak_plot.add_subplot(spec[1,1]) #energy (secondary peak)
    nrg_sec.plot(inp_vars, peak_params[3], label='unabsorbed')
    nrg_sec.plot(inp_vars, peak_params[7], color='red', linestyle=(0, (2, 4)), label='absorbed')
    nrg_sec.legend()
    nrg_sec.set_title('Secondary energy variation')
    nrg_sec.set_ylabel('Secondary energy (keV)')
    nrg_sec.set_xlabel('{} ({})'.format(var_name, unit))
    
    ## difference
    nrg_diff_unabs = peak_params[1]-peak_params[3]
    nrg_diff_abs = peak_params[5]-peak_params[7]
    
    nrg_diff = peak_plot.add_subplot(spec[2,1]) #energy diff
    nrg_diff.plot(inp_vars, nrg_diff_unabs, label='unabsorbed')
    nrg_diff.plot(inp_vars, nrg_diff_abs, color='red', linestyle=(0, (2, 4)), label='absorbed')
    nrg_diff.legend()
    nrg_diff.set_title('Energy difference variation')
    nrg_diff.set_ylabel('Energy difference (keV)')
    nrg_diff.set_xlabel('{} ({})'.format(var_name, unit))
    
    # subplot_showing
    peak_plot

#other xrad2lin Parameters
    other_plot = plt.figure(figsize=(12, 8))                        
    other_plot.suptitle('{} effect on other parameters'.format(var_name))
    spec = mpl.gridspec.GridSpec(ncols = 2, nrows=2)
    spec.update(wspace = 0.25, hspace = 0.25, top = 0.93)
    
    # flux
    ## unabsorbed vs absorbed
    flux = other_plot.add_subplot(spec[0,0]) #flux (main peak)
    flux.plot(inp_vars, xrad2lin_params[0], label='unabsorbed')
    flux.plot(inp_vars, xrad2lin_params[1], color='red', linestyle=(0, (2, 4)), label='absorbed')
    flux.legend()
    flux.set_title('line flux variation')
    flux.set_ylabel('line flux (arbitrary units)')
    flux.set_xlabel('{} ({})'.format(var_name, unit))
    
    ## ratio
    flux_rat = other_plot.add_subplot(spec[1,0]) #flux (main peak)
    flux_rat.plot(inp_vars, xrad2lin_params[2])
    # flux_rat.legend()
    flux_rat.set_title('flux ratio variation')
    flux_rat.set_ylabel('flux ratio ()')
    flux_rat.set_xlabel('{} ({})'.format(var_name, unit))
    

    
    # coverage and rays
    ## rays
    rays = other_plot.add_subplot(spec[0,1]) #flux (main peak)
    rays.plot(inp_vars, xrad2lin_params[3], label='emitted')
    rays.plot(inp_vars, xrad2lin_params[4], color='red', linestyle=(0, (2, 4)), label='absorbed')
    rays.legend()
    rays.set_title('emitted and absorbed rays')
    rays.set_ylabel('number of rays')
    rays.set_xlabel('{} ({})'.format(var_name, unit))
    
    ## coverage ratio
    cover = other_plot.add_subplot(spec[1,1]) #flux (main peak)
    cover.plot(inp_vars, xrad2lin_params[5])
    # flux_rat.legend()
    cover.set_title('coverage ratio variation')
    cover.set_ylabel('coverage ratio (%)')
    cover.set_xlabel('{} ({})'.format(var_name, unit))
    
    # subplot_showing
    other_plot
# ##############################################################################

 
#%% ACTUALLY MAKING THE GRAPH

#%% ## LINE 1 PARAMS
#%% #### viewing angle variations #################################################
angle_var_arr = np.linspace(0.5, 89.5, num=100) #set it to 100... otherwise it's too slow


angle_xrad2lin_params = np.zeros((6, 100))
angle_fwhm = np.zeros((4, 100))
angle_peak_params = np.zeros((8, 100))


for i in tqdm(range(100)): # grabbing the values are not collapsed into procedure to to make it easier to iterate this
    inp_lin1_va_var = "{} {} {}".format(angle_var_arr[i], 
                                        distance_cont, 
                                        zoom_out_fct_cont)
    
    d_xrad2lin_params, d_fwhm, d_peak_params = line_by_line_var(line1 = inp_lin1_va_var)
    
    angle_xrad2lin_params[:, i] = d_xrad2lin_params[:]
    angle_fwhm[:, i] = d_fwhm[:]
    angle_peak_params[:, i] = d_peak_params[:]
    
plotting (angle_var_arr, angle_xrad2lin_params, angle_fwhm, angle_peak_params, var_name='viewing angle', unit='degree')


#%% #### observer distance variations #################################################
dist_var_arr = np.linspace(250, 2500, num=100) #set it to 100... otherwise it's too slow


dist_xrad2lin_params = np.zeros((6, 100))
dist_fwhm = np.zeros((4, 100))
dist_peak_params = np.zeros((8, 100))


for i in tqdm(range(100)): # grabbing the values are not collapsed into procedure to to make it easier to iterate this
    inp_lin1_dist_var = "{} {} {}".format(viewing_angle_cont, 
                                        dist_var_arr[i], 
                                        zoom_out_fct_cont)
    
    d_xrad2lin_params, d_fwhm, d_peak_params = line_by_line_var(line1 = inp_lin1_dist_var)
    
    dist_xrad2lin_params[:, i] = d_xrad2lin_params[:]
    dist_fwhm[:, i] = d_fwhm[:]
    dist_peak_params[:, i] = d_peak_params[:]
    
plotting (dist_var_arr, dist_xrad2lin_params, dist_fwhm, dist_peak_params, var_name='distance', unit='Rg')



#%% # LINE 2 VARS ################################################################
#%% ### angular momentum variations ##############################################
angm_var_arr = np.linspace(0.001, 0.999, num=100) #set it to 100... otherwise it's too slow


angm_xrad2lin_params = np.zeros((6, 100))
angm_fwhm = np.zeros((4, 100))
angm_peak_params = np.zeros((8, 100))


for i in tqdm(range(100)): # grabbing the values are not collapsed into procedure to to make it easier to iterate this
    inp_lin2_angm_var = "{} {} {} {}".format(angm_var_arr[i], 
                                                  R_in_cont, 
                                                  R_out_cont, 
                                                  emissivity_idx_cont)
    
    d_xrad2lin_params, d_fwhm, d_peak_params = line_by_line_var(line2 = inp_lin2_angm_var)
    
    angm_xrad2lin_params[:, i] = d_xrad2lin_params[:]
    angm_fwhm[:, i] = d_fwhm[:]
    angm_peak_params[:, i] = d_peak_params[:]
    
plotting (angm_var_arr, angm_xrad2lin_params, angm_fwhm, angm_peak_params, var_name='angular momentum', unit='')


#%% #### inner radius variations ##############################################
R_in_var_arr = np.linspace(0.1, 15, num=100) #set it to 100... otherwise it's too slow


R_in_xrad2lin_params = np.zeros((6, 100))
R_in_fwhm = np.zeros((4, 100))
R_in_peak_params = np.zeros((8, 100))


for i in tqdm(range(100)): # grabbing the values are not collapsed into procedure to to make it easier to iterate this
    inp_lin2_R_in_var = "{} {} {} {}".format(angular_momentum_cont, 
                                                  R_in_var_arr[i], 
                                                  R_out_cont, 
                                                  emissivity_idx_cont)
    
    d_xrad2lin_params, d_fwhm, d_peak_params = line_by_line_var(line2 = inp_lin2_R_in_var)
    
    R_in_xrad2lin_params[:, i] = d_xrad2lin_params[:]
    R_in_fwhm[:, i] = d_fwhm[:]
    R_in_peak_params[:, i] = d_peak_params[:]
    
plotting (R_in_var_arr, R_in_xrad2lin_params, R_in_fwhm, R_in_peak_params, var_name='inner radius', unit='Rg')


#%% #### outer radius variations ##############################################
R_out_var_arr = np.linspace(5, 100, num=100) #set it to 100... otherwise it's too slow


R_out_xrad2lin_params = np.zeros((6, 100))
R_out_fwhm = np.zeros((4, 100))
R_out_peak_params = np.zeros((8, 100))


for i in tqdm(range(100)): # grabbing the values are not collapsed into procedure to to make it easier to iterate this
    inp_lin2_R_out_var = "{} {} {} {}".format(angular_momentum_cont, 
                                                  R_in_cont, 
                                                  R_out_var_arr[i], 
                                                  emissivity_idx_cont)
    
    d_xrad2lin_params, d_fwhm, d_peak_params = line_by_line_var(line2 = inp_lin2_R_out_var)
    
    R_out_xrad2lin_params[:, i] = d_xrad2lin_params[:]
    R_out_fwhm[:, i] = d_fwhm[:]
    R_out_peak_params[:, i] = d_peak_params[:]
    
plotting (R_out_var_arr, R_out_xrad2lin_params, R_out_fwhm, R_out_peak_params, var_name='outer radius', unit='Rg')



#%% #### emissivity index variations ##############################################
em_var_arr = np.linspace(0.1, 5, num=100) #set it to 100... otherwise it's too slow


em_xrad2lin_params = np.zeros((6, 100))
em_fwhm = np.zeros((4, 100))
em_peak_params = np.zeros((8, 100))


for i in tqdm(range(100)): # grabbing the values are not collapsed into procedure to to make it easier to iterate this
    inp_lin2_em_var = "{} {} {} {}".format(angular_momentum_cont, 
                                                  R_in_cont, 
                                                  R_out_cont, 
                                                  em_var_arr[i])
    
    d_xrad2lin_params, d_fwhm, d_peak_params = line_by_line_var(line2 = inp_lin2_em_var)
    
    em_xrad2lin_params[:, i] = d_xrad2lin_params[:]
    em_fwhm[:, i] = d_fwhm[:]
    em_peak_params[:, i] = d_peak_params[:]
    
plotting (em_var_arr, em_xrad2lin_params, em_fwhm, em_peak_params, var_name='emissivity index', unit='')



#%% # LINE 3 VARIATION ############################################################
#%% ### absorber number variation ##############################################
absnum_var_arr = np.linspace(1, 1500, num=100) 
absnum_var_arr[:] = np.floor(absnum_var_arr[:])
absnum_var_arr = np.array(absnum_var_arr).astype(int).tolist()
#has to be an integer (Think it need to be converted with int() fct)


absnum_xrad2lin_params = np.zeros((6, 100))
absnum_fwhm = np.zeros((4, 100))
absnum_peak_params = np.zeros((8, 100))


for i in tqdm(range(100)): # grabbing the values are not collapsed into procedure to to make it easier to iterate this
    inp_lin3_absnum_var = "{} {} {} {} {} {}".format(seed, 
                                                absnum_var_arr[i],
                                                abs_rad_cont,
                                                center_x_coord_cont,
                                                center_y_coord_cont,
                                                std_dev_cont)
    
    d_xrad2lin_params, d_fwhm, d_peak_params = line_by_line_var(line3 = inp_lin3_absnum_var)
    
    absnum_xrad2lin_params[:, i] = d_xrad2lin_params[:]
    absnum_fwhm[:, i] = d_fwhm[:]
    absnum_peak_params[:, i] = d_peak_params[:]
    
plotting (absnum_var_arr, absnum_xrad2lin_params, absnum_fwhm, absnum_peak_params, var_name='absorber number', unit='')


#%% #### absorber radius variations ##############################################
absrad_var_arr = np.linspace(0.1, 5, num=100) #for some reason, bug in the code prevents radius to be more than 5 (don't know what's causing this...) expectation: too high on the absorbtion (?) encountering negative value (?)


absrad_xrad2lin_params = np.zeros((6, 100))
absrad_fwhm = np.zeros((4, 100))
absrad_peak_params = np.zeros((8, 100))


for i in tqdm(range(100)): # grabbing the values are not collapsed into procedure to to make it easier to iterate this
    inp_lin3_absrad_var = "{} {} {} {} {} {}".format(seed, 
                                                num_abs_cont,
                                                absrad_var_arr[i],
                                                center_x_coord_cont,
                                                center_y_coord_cont,
                                                std_dev_cont)
    
    d_xrad2lin_params, d_fwhm, d_peak_params = line_by_line_var(line3 = inp_lin3_absrad_var)
    
    absrad_xrad2lin_params[:, i] = d_xrad2lin_params[:]
    absrad_fwhm[:, i] = d_fwhm[:]
    absrad_peak_params[:, i] = d_peak_params[:]
    
plotting (absrad_var_arr, absrad_xrad2lin_params, absrad_fwhm, absrad_peak_params, var_name='absorber radius', unit='Rg')
#%% ###########################################################################



#%%

# ## Extra Graphs
# ### angle at 70 degrees, search for no secondary peak

# viewing_angle_70deg = 30.
# distance_cont = 1000.
# zoom_out_fct_cont = 1.

# inp_lin1_70deg = "{} {} {}".format(viewing_angle_70deg, 
#                                   distance_cont, 
#                                   zoom_out_fct_cont)


# deg70_energy, deg70_flux_unabs, deg70_flux_abs = xrad2run(line1 = inp_lin1_70deg)

# deg70_fwhm_line = fwhm_calc(deg70_energy, deg70_flux_unabs, deg70_flux_abs, graph='True')


# ### angular momentum near 0.8 at 70 degrees, search for no secondary peak

# angular_momentum_AM = 0.9
# R_in_cont = .1
# R_out_cont = 19.
# emissivity_idx_cont = 2.5

# inp_lin2_AM = "{} {} {} {}".format(angular_momentum_AM, 
#                                      R_in_cont, 
#                                      R_out_cont, 
#                                      emissivity_idx_cont)


# am_energy, am_flux_unabs, am_flux_abs = xrad2run(line2 = inp_lin2_AM)

# am_fwhm_line = fwhm_calc(am_energy, am_flux_unabs, am_flux_abs, graph='True')




## absorbers

# seed = -13 #does not change
# num_abs_cont = 1
# abs_rad_var = 5
# center_x_coord_cont = 0.
# center_y_coord_cont = 0.
# std_dev_cont = 7.

# inp_lin3_abrad = "{} {} {} {} {} {}".format(seed, 
#                                             num_abs_cont,
#                                             abs_rad_var,
#                                             center_x_coord_cont,
#                                             center_y_coord_cont,
#                                             std_dev_cont)


# absrad_energy, absrad_flux_unabs, absrad_flux_abs = xrad2run(line3 = inp_lin3_abrad)

# am_fwhm_line = fwhm_calc(absrad_energy, absrad_flux_unabs, absrad_flux_abs, graph='True')

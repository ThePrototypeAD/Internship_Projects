# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 05:21:47 2024

@author: deand
"""

import sys
import os
import glob

import math
import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from pathlib import Path
from tqdm import tqdm

import astropy.io.fits as fits
import scipy.constants as const

path = r'C:\__Programming\__MassS3\Internship\__QSO_candidate' 
os.chdir(path) #change the directory

science_obj_list = pd.read_csv("science_object_list.csv")

science_sdss_name = science_obj_list.SDSS_NAME
science_sdss_name = np.array(science_sdss_name, dtype='str_') #convert into array of string

science_Z = science_obj_list.Z
science_Z = np.array(science_Z)
###############################################################################



# PRELIMINARY FUNCTIONS #######################################################
## rest wavelength function ###################################################
"""
since sdss wavelength are the observed wavelength, conversion to rest must be 
applied. With the redshift information, this can be done.

input:  z -> as obtained from the DR16 QSO catalogue (float)
       lambda_obs -> lambda as observed from SDSS (array of float) (in angstrom)

output: lambda_rest -> rest wavelength (array of float)  (in angstrom)
"""

def rest_wavelength_fct (z, lambda_obs): 
    
    lambda_rest = lambda_obs / (z+1)
    return lambda_rest

## velocity shift #############################################################
"""
reflecting from Eracleous presentation, wavelength shift are represented in km/s.
the following function converts delta lambda into velocity. inputs have to be 
in same unit

input:  lambda_convert = wavelength to be converted (array of float)
        lambda_target = target wavelength, the v=0 point (float)
        
output: vel = shift velocity (array of float) (in km/s)
"""

def shift_vel (lambda_convert, lambda_target):
    
    lightspeed_convert = const.c / 1000. #conversion of lightspeed in m/s to km/s
    
    delta_lambda = lambda_convert, lambda_target
    
    z = delta_lambda/lambda_target
    
    return z*lightspeed_convert

###############################################################################




# TAKING LAMBDA, FLUX, SHIFT VELOCITY, OBJECT NAME, AND Z #####################
"""
all of these four parameters will be saved in a dataframe (either pd or fits)
((most likely fits since lambda, flux, and shift velocity columns are arrays))
"""

def downloaded_files_info (folder_name, column_capital = 'False'): #apparently, all v_13_0 flux data are capital
    
    path_dir = Path(r'C:\__Programming\__MassS3\Internship\__QSO_candidate\Spectras\{}'.format(folder_name))
    item_list = os.listdir(path_dir)
    
    output_SDSS_name = []
    output_Z = []
    
    output_lambda_obs = []
    output_lambda_res = []
    output_flux = []
    
## grab the plate, mjd, and fiberID of downloaded files to obtain the z and object name
    for item in tqdm(item_list):
        if (len(item)==26): #this is for 5 digits plate
        
            dummy_plate = np.int32(item[5:10])
            dummy_mjd = np.int32(item[11:16])
            dummy_fiberID = np.int32(item[17:21])
        
        elif (len(item)==25): #this is for 4 digits plate
            dummy_plate = np.int32(item[5:9])
            dummy_mjd = np.int32(item[10:15])
            dummy_fiberID = np.int32(item[16:20])
                
        dummy_n = np.int0(science_obj_list.index[(science_obj_list['PLATE'] == dummy_plate)
                                                & (science_obj_list['MJD'] == dummy_mjd)
                                                & (science_obj_list['FIBERID'] == dummy_fiberID)])  #for some reason, it gives an array...
        
        
        dummy_SDSS_name = (science_sdss_name[dummy_n])
        dummy_SDSS_name = str(dummy_SDSS_name[0])
        
        dummy_Z = (science_Z[dummy_n])
        dummy_Z = dummy_Z[0]
        
## grab the lambda and flux

        spectrum_data = fits.getdata(path_dir / item)
        
        if column_capital == 'True' :
            dummy_loglam_obs = spectrum_data.LOGLAM
            dummy_flux = spectrum_data.FLUX
        else :
            dummy_loglam_obs = spectrum_data.loglam
            dummy_flux = spectrum_data.flux   
        
        dummy_lambda_obs = np.power(10, dummy_loglam_obs)
        dummy_lambda_res = rest_wavelength_fct(dummy_Z, dummy_lambda_obs)
        
        
## putting it all together as an array
        output_SDSS_name.append(dummy_SDSS_name)
        output_Z.append(dummy_Z)
        
        output_flux.append(dummy_flux)
        output_lambda_res.append(dummy_lambda_res)
        output_lambda_obs.append(dummy_lambda_obs)
    
    return output_SDSS_name, output_Z, output_flux, output_lambda_res, output_lambda_obs


SDSS_name_103, Z_103, flux_103, lambda_res_103, lambda_obs_103 = downloaded_files_info("103")
SDSS_name_26, Z_26, flux_26, lambda_res_26, lambda_obs_26 = downloaded_files_info("26")
SDSS_name_v5, Z_v5, flux_v5, lambda_res_v5, lambda_obs_v5 = downloaded_files_info("v5_13_0", column_capital='True')

# combine into one
zzcombo_SDSS_name = np.concatenate((SDSS_name_26, SDSS_name_103, SDSS_name_v5))
zzcombo_Z = np.concatenate((Z_26, Z_103, Z_v5))

zzcombo_flux = np.concatenate((flux_26, flux_103, flux_v5), axis=0)
zzcombo_lambda_res = np.concatenate((lambda_res_26, lambda_res_103, lambda_res_v5), axis=0)
zzcombo_lambda_obs = np.concatenate((lambda_obs_26, lambda_obs_103, lambda_obs_v5), axis=0)


#saving as fits file
# save_SDSS_name = fits.Column(name='SDSS_name', array=SDSS_name_103, format='A')
# save_Z = fits.Column(name='Z', array=Z_103, format='E')
# save_flux = fits.Column(name='Flux', array=flux_103, format='P')
# save_lambda_res = fits.Column(name='Rest Wavelength', array=lambda_res_103, format='P')
# save_lambds_obs = fits.Column(name='Observed Wavelength', array=lambda_obs_103, format='P')

# cols = fits.ColDefs([save_SDSS_name, save_Z, save_flux, save_lambda_res, save_lambds_obs])

# hdu_save = fits.BinTableHDU.from_columns(cols)

# hdu_save.writeto('save_data.fits')

#SCREW IT, SAVE IT AS ONE BY ONE













# if (len('spec-3586-55181-0098.fits')==26): #this is for 5 digits plate
        
#     dummy_plate = np.int32('spec-3586-55181-0098.fits'[5:10])
#     dummy_mjd = np.int32('spec-3586-55181-0098.fits'[11:16])
#     dummy_fiberID = np.int32('spec-3586-55181-0098.fits'[17:21])
        
# elif (len('spec-3586-55181-0098.fits')==25): #this is for 4 digits plate
#     dummy_plate = np.int32('spec-3586-55181-0098.fits'[5:9])
#     dummy_mjd = np.int32('spec-3586-55181-0098.fits'[10:15])
#     dummy_fiberID = np.int32('spec-3586-55181-0098.fits'[16:20])
                
# dummy_n = np.int0(science_obj_list.index[(science_obj_list['PLATE'] == dummy_plate)
#                                                 & (science_obj_list['MJD'] == dummy_mjd)
#                                                 & (science_obj_list['FIBERID'] == dummy_fiberID)])  #for some reason, it gives an array...
        
        
# dummy_SDSS_name = (science_sdss_name[dummy_n])
# dummy_SDSS_name = str(dummy_SDSS_name[0])
# dummy_Z = (science_Z[dummy_n])


# path_dir = Path(r'C:\__Programming\__MassS3\Internship\__QSO_candidate\Spectras\v5_13_0') 
# spectrum_data = fits.getdata(path_dir / 'spec-3586-55181-0098.fits')

# if spectrum_data.loglam :
#     dummy_loglam_obs = spectrum_data.loglam
#     dummy_flux = spectrum_data.flux        
        
# elif spectrum_data.LOGLAM :
#     dummy_loglam_obs = spectrum_data.LOGLAM
#     dummy_flux = spectrum_data.FLUX
        
# dummy_lambda_obs = np.power(10, dummy_loglam_obs)
# dummy_lambda_res = rest_wavelength_fct(dummy_Z, dummy_lambda_obs)





        
        





# path_dir = Path(r'C:\__Programming\__MassS3\Internship\__QSO_candidate\Spectras\26') 
# item_list = os.listdir(path_dir)

# test = fits.getdata(path_dir / 'spec-0266-51630-0003.fits')

# lambda_obs = np.power(10, test.loglam)


## note: run on first 100 data in each folders
## also note: grab the ra and declination
        ## correct for reddening by RA and declination (confirm with the paper again) (galactic exinction correction)
        ## use Fantasy code
        

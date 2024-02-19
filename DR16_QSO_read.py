# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:29:41 2024

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

from pathlib import Path
from tqdm import tqdm

import astropy.io.fits as fits

path = r'C:\__Programming\__MassS3\Internship\__QSO_candidate' 
os.chdir(path) #change the directory
###############################################################################



#%%
##### OPENING THE MAIN FILE AND QUERYING THE Z<0.8 OBJECTS ###################

# DR16_v3_superset = fits.open("DR16Q_Superset_v3.fits")
DR16_v4_set = fits.open("DR16Q_v4.fits")

#v3_qso_list = DR16_v3_superset[1].data      #v3 = all observations   
v4_qso_list = DR16_v4_set[1].data           #v4 = final qso only catalog



############# CONVERSION TO PANDAS DATAFRAME #################################
# v3_qso_list = pd.DataFrame.from_records(v3_qso_list)
    # this can't be done as some per-colum arrays are not 1D
## STATUS: FAILURE -> proceed to directly access the records##################


#check for negative redshift
# v3_neg_redshift = np.where(v3_qso_list.Z < 0.)
v4_neg_redshift = np.where(v4_qso_list.Z < 0.)      #result in 10 objects with negative redshift. Q: what does this means?
                                                    #further filter it for now, remove from all index


# #check index, grab index for objects with z<0.8 (OOI = object of interest - science obj)
v4_ooi_index = np.transpose(np.array(np.where((v4_qso_list.Z < 0.8) & (v4_qso_list.Z > 0.0))))    
    #result in around 76 565 objects, with 10 objects have negative redshift (compare to Eracleous 15,900 objects)
    #   10/76 565 = 0.013% data ignored
    # possible next science, what's the deal with negative redshift objects? errors?


### grabbing the object into different array
v4_ooi_list = np.zeros_like(v4_qso_list)
v4_ooi_list.resize(len(v4_ooi_index))

science_names_1 = []

for i in tqdm(range(len(v4_ooi_index))):        #tqdm for progress bar
    grabbed_index = int(v4_ooi_index[i])        
        #for some reason need to convert to int...
        #prolly should do this when grabbing it
    v4_ooi_list[i] = v4_qso_list[grabbed_index]
    science_names_1.append(v4_qso_list.SDSS_NAME[i])

# # create a dataframe from it, grab the object name, plate, mjd, fiberID
science_names = v4_ooi_list.SDSS_NAME
science_plate = v4_ooi_list.PLATE
science_mjd = v4_ooi_list.MJD
science_fiberID = v4_ooi_list.FIBERID
science_Z = v4_ooi_list.Z
science_RA = v4_ooi_list.RA
science_DEC = v4_ooi_list.DEC

v4_science_selected_params = pd.DataFrame({'SDSS_NAME': science_names_1,
                                           'RA': science_RA,
                                           'DEC': science_DEC,
                                            'PLATE': science_plate,
                                            'MJD': science_mjd,
                                            'FIBERID': science_fiberID,
                                            'Z': science_Z})

v4_science_selected_params.to_csv('science_object_list.csv')

# # saving the file array for future reference
# hdu = fits.BinTableHDU.from_columns(v4_ooi_list) #for some reason, this is the v4_qso_list...
# hdu.writeto('science_object_list.fits')

###############################################################################




#%%
###### MAKING THE LIST TO DOWNLOAD THE SPECTRA VIA TERMINAL ###################

#it seems they don't have the spectra here... I guess it's time to grab them individually
# #load the saved v4_ooi_list array to bypass the need of running the above script every runs
# some trickery is done, saved and opened

v4_ooi_list = pd.read_csv("science_object_list.csv")

science_sdss_names = v4_ooi_list['SDSS_NAME']
science_RA = v4_ooi_list['RA']
science_DEC = v4_ooi_list['DEC']
science_plate = v4_ooi_list['PLATE']
science_mjd = v4_ooi_list['MJD']
science_fiberID = v4_ooi_list['FIBERID']

plate = v4_ooi_list.PLATE
mjd = v4_ooi_list.MJD
fiberID = v4_ooi_list.FIBERID

#grab spectra from DR18
link_path_103_DR18 = []
link_path_104_DR18 = []
link_path_26_DR18 = []
link_path_v5_13_2_DR18 = []
link_path_v6_0_4_DR18 = []

#grab spectra from DR16
link_path_103_DR16 = []
link_path_104_DR16 = []
link_path_26_DR16 = []
link_path_v5_13_0_DR16 = []


for n in tqdm(range(len(plate))):
    #from DR18
    link_path_103_DR18.append('https://data.sdss.org/sas/dr18/spectro/sdss/redux/103/spectra/lite/{}/spec-{}-{}-{}.fits'.format(f"{(plate[n]):04d}", f"{(plate[n]):04d}", mjd[n], f"{(fiberID)[n]:04d}"))
    link_path_104_DR18.append('https://data.sdss.org/sas/dr18/spectro/sdss/redux/104/spectra/lite/{}/spec-{}-{}-{}.fits'.format(f"{(plate[n]):04d}", f"{(plate[n]):04d}", mjd[n], f"{(fiberID)[n]:04d}"))
    link_path_26_DR18.append('https://data.sdss.org/sas/dr18/spectro/sdss/redux/26/spectra/lite/{}/spec-{}-{}-{}.fits'.format(f"{(plate[n]):04d}", f"{(plate[n]):04d}", mjd[n], f"{(fiberID)[n]:04d}"))
    link_path_v5_13_2_DR18.append('https://data.sdss.org/sas/dr18/spectro/sdss/redux/v5_13_2/spectra/lite/{}/spec-{}-{}-{}.fits'.format(f"{(plate[n]):04d}", f"{(plate[n]):04d}", mjd[n], f"{(fiberID)[n]:04d}"))
    link_path_v6_0_4_DR18.append('https://data.sdss.org/sas/dr18/spectro/sdss/redux/v6_0_4/spectra/lite/{}/spec-{}-{}-{}.fits'.format(f"{(plate[n]):04d}", f"{(plate[n]):04d}", mjd[n], f"{(fiberID)[n]:04d}"))
    
    #from DR16
    link_path_103_DR16.append('https://data.sdss.org/sas/dr16/sdss/spectro/redux/103/spectra/lite/{}/spec-{}-{}-{}.fits'.format(f"{(plate[n]):04d}", f"{(plate[n]):04d}", mjd[n], f"{(fiberID)[n]:04d}"))
    link_path_104_DR16.append('https://data.sdss.org/sas/dr16/sdss/spectro/redux/104/spectra/lite/{}/spec-{}-{}-{}.fits'.format(f"{(plate[n]):04d}", f"{(plate[n]):04d}", mjd[n], f"{(fiberID)[n]:04d}"))
    link_path_26_DR16.append('https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/lite/{}/spec-{}-{}-{}.fits'.format(f"{(plate[n]):04d}", f"{(plate[n]):04d}", mjd[n], f"{(fiberID)[n]:04d}"))
    link_path_v5_13_0_DR16.append('https://data.sdss.org/sas/dr16/sdss/spectro/redux/v5_13_0/spectra/lite/{}/spec-{}-{}-{}.fits'.format(f"{(plate[n]):04d}", f"{(plate[n]):04d}", mjd[n], f"{(fiberID)[n]:04d}"))
    

link_path_all_DR18 = link_path_103_DR18 + link_path_104_DR18 + link_path_26_DR18 + link_path_v5_13_2_DR18 + link_path_v6_0_4_DR18
link_path_all_DR16 = link_path_103_DR16 + link_path_104_DR16 + link_path_26_DR16 + link_path_v5_13_0_DR16

#exporting to .txt file
np.savetxt('__DR18_103_download.txt', np.array(link_path_103_DR18), fmt='%s')
np.savetxt('__DR18_104_download.txt', np.array(link_path_104_DR18), fmt='%s')
np.savetxt('__DR18_26_download.txt', np.array(link_path_26_DR18), fmt='%s')
np.savetxt('__DR18_v5_13_2_download.txt', np.array(link_path_v5_13_2_DR18), fmt='%s')
np.savetxt('__DR18_v6_0_4_download.txt', np.array(link_path_v6_0_4_DR18), fmt='%s')
np.savetxt('__DR18__all_download.txt', np.array(link_path_all_DR18), fmt='%s')

np.savetxt('__DR16_103_download.txt', np.array(link_path_103_DR16), fmt='%s')
np.savetxt('__DR16_104_download.txt', np.array(link_path_104_DR16), fmt='%s')
np.savetxt('__DR16_26_download.txt', np.array(link_path_26_DR16), fmt='%s')
np.savetxt('__DR16_v5_13_0_download.txt', np.array(link_path_v5_13_0_DR16), fmt='%s')
np.savetxt('__DR16__all_download.txt', np.array(link_path_all_DR16), fmt='%s')

###############################################################################


#check the address, especially the 104
#check if the error is from the url writing, or not


#%%
############## VERIFYING THE FILES  ###########################################
## checking which files are not downloaded

## grab the plate, mjd, and fiberID of downloaded files
def check_downloaded_files (folder_name):
    output_plate = []
    output_mjd = []
    output_fiberID = []
    
    path_dir = Path(r'C:\__Programming\__MassS3\Internship\__QSO_candidate\Spectras\{}'.format(folder_name)) 
    item_list = os.listdir(path_dir)
    
    for item in tqdm(item_list):
        if (len(item)==26): #this is for 5 digits plate
        
            dummy_plate = np.int32(item[5:10])
            dummy_mjd = np.int32(item[11:16])
            dummy_fiberID = np.int32(item[17:21])
        
        elif (len(item)==25): #this is for 4 digits plate
            dummy_plate = np.int32(item[5:9])
            dummy_mjd = np.int32(item[10:15])
            dummy_fiberID = np.int32(item[16:20])
                
        output_plate.append(dummy_plate)
        output_mjd.append(dummy_mjd)
        output_fiberID.append(dummy_fiberID)    
    
    return output_plate, output_mjd, output_fiberID

plate_26, mjd_26, fiberID_26 = check_downloaded_files("26")
plate_103, mjd_103, fiberID_103 = check_downloaded_files("103")
plate_104, mjd_104, fiberID_104 = check_downloaded_files("104")
plate_v5, mjd_v5, fiberID_v5 = check_downloaded_files("V5_13_0")

plate_all = np.concatenate((plate_26, plate_103, plate_104, plate_v5))
mjd_all = np.concatenate((mjd_26, mjd_103, mjd_104, mjd_v5))
fiberID_all = np.concatenate((fiberID_26, fiberID_103, fiberID_104, fiberID_v5))


n = [] #index for data
for i in tqdm(range(len(plate_all))):
    dummy_n = np.int0(v4_ooi_list.index[(v4_ooi_list['PLATE'] == plate_all[i])
                                    & (v4_ooi_list['MJD'] == mjd_all[i])
                                    & (v4_ooi_list['FIBERID'] == fiberID_all[i])])  #for some reason, it gives an array...
    n.append(dummy_n[0])



unavailable_objects=[]
unavailable_plate = []
unavailable_mjd = []
unavailable_fiberID = []

for i in tqdm(range(len(v4_ooi_list))):
    if (i not in n):
        unavailable_objects.append(science_sdss_names[i])
        unavailable_plate.append(science_plate[i])
        unavailable_mjd.append(science_mjd[i])
        unavailable_fiberID.append(science_fiberID[i])



# # check if really unavailable in dr16
link_path_103_DR16_xx = []
link_path_104_DR16_xx = []
link_path_26_DR16_xx = []
link_path_v5_13_0_DR16_xx = []

for n in tqdm(range(len(unavailable_plate))):

    #from DR16
    link_path_103_DR16_xx.append('https://data.sdss.org/sas/dr16/sdss/spectro/redux/103/spectra/lite/{}/spec-{}-{}-{}.fits'.format(f"{(unavailable_plate[n]):04d}", f"{(unavailable_plate[n]):04d}", unavailable_mjd[n], f"{(unavailable_fiberID)[n]:04d}"))
    link_path_104_DR16_xx.append('https://data.sdss.org/sas/dr16/sdss/spectro/redux/104/spectra/lite/{}/spec-{}-{}-{}.fits'.format(f"{(unavailable_plate[n]):04d}", f"{(unavailable_plate[n]):04d}", unavailable_mjd[n], f"{(unavailable_fiberID)[n]:04d}"))
    link_path_26_DR16_xx.append('https://data.sdss.org/sas/dr16/sdss/spectro/redux/26/spectra/lite/{}/spec-{}-{}-{}.fits'.format(f"{(unavailable_plate[n]):04d}", f"{(unavailable_plate[n]):04d}", unavailable_mjd[n], f"{(unavailable_fiberID)[n]:04d}"))
    link_path_v5_13_0_DR16_xx.append('https://data.sdss.org/sas/dr16/sdss/spectro/redux/v5_13_0/spectra/lite/{}/spec-{}-{}-{}.fits'.format(f"{(unavailable_plate[n]):04d}", f"{(unavailable_plate[n]):04d}", unavailable_mjd[n], f"{(unavailable_fiberID)[n]:04d}"))
    

link_path_all_DR16_unavail = link_path_103_DR16_xx + link_path_104_DR16_xx + link_path_26_DR16_xx + link_path_v5_13_0_DR16_xx

np.savetxt('__all_download_unavailable.txt', np.array(link_path_all_DR16_unavail), fmt='%s')

###############################################################################




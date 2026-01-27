# use Group 1 Spectrum Code/SpectrumGenerator.py to read the csv file PandExoParametrers and update the parameters below accordingly



import warnings
warnings.filterwarnings('ignore')
import pandexo.engine.justdoit as jdi # THIS IS THE HOLY GRAIL OF PANDEXO
import numpy as np
import os
exo_dict = jdi.load_exo_dict()
exo_dict['observation']['sat_level'] = 80    #saturation level in percent of full well 
exo_dict['observation']['sat_unit'] = '%' 
exo_dict['observation']['noccultations'] = 2 #number of transits 
exo_dict['observation']['R'] = None          #fixed binning. I usually suggest ZERO binning.. you can always bin later 
                                             #without having to redo the calcualtion
exo_dict['observation']['baseline'] = 1.0    #fraction of time in transit versus out = in/out
exo_dict['observation']['baseline_unit'] = 'frac' 
exo_dict['observation']['noise_floor'] = 0   #this can be a fixed level or it can be a filepath 
exo_dict['star']['type'] = 'phoenix'        #phoenix or user (if you have your own)
exo_dict['star']['mag'] = 8.0               #magnitude of the system
exo_dict['star']['ref_wave'] = 1.25         #For J mag = 1.25, H = 1.6, K =2.22.. etc (all in micron)
exo_dict['star']['temp'] = 5500             #in K 
exo_dict['star']['metal'] = 0.0             # as log Fe/H
exo_dict['star']['logg'] = 4.0
exo_dict['star']['radius'] = 1
exo_dict['star']['r_unit'] = 'R_sun'    
exo_dict['planet']['type'] = 'constant'
exo_dict['planet']['radius'] = 1                      #other options include "um","nm" ,"Angs", "secs" (for phase curves)
exo_dict['planet']['r_unit'] = 'R_jup'  
exo_dict['planet']['transit_duration'] = 2.0*60.0*60.0 
exo_dict['planet']['td_unit'] = 's'
exo_dict['planet']['f_unit'] = 'rp^2/r*^2'
print('Starting TEST run')
jdi.run_pandexo(exo_dict, ['NIRSpec G140H'], save_file=False)
print('SUCCESS') 
import pandexo.engine.justplotit as jpi 
import pickle as pk


exo_dict = jdi.load_exo_dict()

exo_dict['star']['type'] = 'phoenix'      
exo_dict['star']['temp'] = 3101 
exo_dict['star']['metal'] =   0.24             ## as log Fe/H
exo_dict['star']['logg'] =  5.0286            ## log gravity cgs
exo_dict['star']['mag'] = 9.75
exo_dict['star']['ref_wave'] = 1.25        #For J mag = 1.25, H = 1.6, K =2.22.. etc (all in micron) 
exo_dict['star']['radius'] = 0.2162
exo_dict['star']['r_unit'] = 'R_sun'
exo_dict['planet']['radius'] = 0.24382235
exo_dict['planet']['r_unit'] = 'R_jup'     ## or R_earth

exo_dict['planet']['transit_duration'] = 0.036235833333333335
exo_dict['planet']['td_unit'] = 'd'

exo_dict['planet']['type'] = 'constant'  ## 'constant', 'user'
##exo_dict['planet']['exopath'] = 'wasp12b.txt'
exo_dict['planet']['f_unit'] = 'rp^2/r*^2'
#exo_dict['planet']['w_unit'] = 'um'. # wavelength unit for user defined spectra

exo_dict['observation']['baseline'] = 1.0 
exo_dict['observation']['baseline_unit'] = 'frac'
exo_dict['observation']['noccultations'] = 1         ## number of transits

inst_dict = jdi.load_mode_dict('NIRSpec G235H')        ## Choose instrument/mode
#inst_dict["configuration"]["detector"]["subarray"] = 'substrip256'  ##to change the sub array
exo_dict['observation']['sat_level'] = 100    #saturation level in percent of full well 
exo_dict['observation']['sat_unit'] = '%' 
exo_dict['observation']['noise_floor'] = 0

result = jdi.run_pandexo(exo_dict, inst_dict, save_file=False) ## save file = 'test.p'

x,y, e = jpi.jwst_1d_spec(result, R=100, num_tran=1, model=False, x_range=[1.6,3.0])
data = jpi.jwst_2d_det(result)

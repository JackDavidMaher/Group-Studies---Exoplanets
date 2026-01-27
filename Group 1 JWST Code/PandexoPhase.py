# use Group 1 Spectrum Code/SpectrumGenerator.py to read the csv file PandExoParametrers and update the parameters below accordingly

import os
import warnings
warnings.filterwarnings('ignore')

import csv
import pandexo.engine.justdoit as jdi 
import pandexo.engine.justplotit as jpi 

import numpy as np
import pickle as pk
import scipy.constants as sc
from astropy import constants as const


# This section reads the PandExoParameters.csv file and extracts parameters
with open("Group 1 Spectrum Code/PandExoParameters.csv", newline="") as PandExoParametersFile:
	reader = csv.reader(PandExoParametersFile)
	header = next(reader)
	data = []
	planetNames = []
	for row in reader:
		row = [item.strip() for item in row]
		if len(row) == 0:
			continue
		*nums, name = row
		conv = [float(item) if item != "" else np.nan for item in nums]
		data.append(conv)
		planetNames.append(name)
	PandExoParameters = np.array(data, dtype=float)

rowCount=0
while rowCount < len(PandExoParameters):    
	PName = planetNames[rowCount]  # Planet name (string)   
	rowCount=rowCount+1

	exo_dict = jdi.load_exo_dict()
    ## star dict
	exo_dict['star']['type'] = 'phoenix'      
	exo_dict['star']['temp'] = PandExoParameters[rowCount][]          ## Temperature in K 
	exo_dict['star']['metal'] = PandExoParameters[rowCount][]         ## Metallacity as log Fe/H
	exo_dict['star']['logg'] =  PandExoParameters[rowCount][]         ## log gravity cgs
	exo_dict['star']['mag'] = PandExoParameters[rowCount][]           ## Star magnitude
	if PandExoParameters[rowCount][] == J:                            ## if loop for the correct magnitude system
		exo_dict['star']['ref_wave'] = 1.25 
	elif PandExoParameters[rowCount][] == K:
		exo_dict['star']['ref_wave'] = 2.22
	else:
		exo_dict['star']['ref_wave'] = 1.60
	exo_dict['star']['radius'] = PandExoParameters[rowCount][]        ##radius of the star in solar radii
	exo_dict['star']['r_unit'] = 'R_sun'

    ## planet dict
	exo_dict['planet']['radius'] = PandExoParameters[rowCount][]     ##radius of the planet in earth radii        
	exo_dict['planet']['r_unit'] = 'R_earth'     ## or R_earth
	exo_dict['planet']['transit_duration'] = PandExoParameters[rowCount][]  ##transit duration in days
	exo_dict['planet']['td_unit'] = 'd'
	exo_dict['planet']['type'] = 'user'            ## 'user' for user defined spectrum
	exo_dict['planet']['exopath'] = f'{}.txt'              ## path to user defined spectrum file
	exo_dict['planet']['f_unit'] = 'rp^2/r*^2'.     ## flux unit for user defined spectrum
	exo_dict['planet']['w_unit'] = 'um'             ## wavelength unit for user defined spectra

	exo_dict['observation']['baseline'] = 1.0 
	exo_dict['observation']['baseline_unit'] = 'frac'
	exo_dict['observation']['noccultations'] = PandExoParameters[rowCount][]  ## number of transits 
	exo_dict['observation']['sat_level'] = 100            #saturation level in percent of full well 
	exo_dict['observation']['sat_unit'] = '%' 
	exo_dict['observation']['noise_floor'] = 0

	result = jdi.run_pandexo(exo_dict, ['NIRSpec G395M'], save_file=True, output_file=f'{PName}.p')
	
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
with open("Group 1 JWST Code/PandExoParameters.csv", newline="") as PandExoParametersFile:
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
	exo_dict['star']['type'] = PandExoParameters[rowCount][8]      
	exo_dict['star']['temp'] = PandExoParameters[rowCount][11]          ## Temperature in K 
	exo_dict['star']['metal'] = PandExoParameters[rowCount][12]         ## Metallacity as log Fe/H
	exo_dict['star']['logg'] =  PandExoParameters[rowCount][13]         ## log gravity cgs
	exo_dict['star']['mag'] = PandExoParameters[rowCount][9]           ## Star magnitude
	exo_dict['star']['ref_wave'] = PandExoParameters[rowCount][10]
	exo_dict['star']['radius'] = PandExoParameters[rowCount][14]        ##radius of the star in solar radii
	exo_dict['star']['r_unit'] = PandExoParameters[rowCount][15]

    ## planet dict
	exo_dict['planet']['radius'] = PandExoParameters[rowCount][17]     ##radius of the planet in earth radii        
	exo_dict['planet']['r_unit'] = PandExoParameters[rowCount][18]     ## or R_earth
	exo_dict['planet']['transit_duration'] = PandExoParameters[rowCount][19]  ##transit duration in days
	exo_dict['planet']['td_unit'] = PandExoParameters[rowCount][20]
	exo_dict['planet']['type'] = PandExoParameters[rowCount][16]            ## 'user' for user defined spectrum
	exo_dict['planet']['exopath'] = f'{}.txt'              ## path to user defined spectrum file
	exo_dict['planet']['f_unit'] = PandExoParameters[rowCount][21]     ## flux unit for user defined spectrum
	exo_dict['planet']['w_unit'] = 'um'             ## wavelength unit for user defined spectra

	exo_dict['observation']['baseline'] = PandExoParameters[rowCount][5]
	exo_dict['observation']['baseline_unit'] = PandExoParameters[rowCount][6]
	exo_dict['observation']['noccultations'] = PandExoParameters[rowCount][3]  ## number of transits 
	exo_dict['observation']['sat_level'] = PandExoParameters[rowCount][1]            #saturation level in percent of full well 
	exo_dict['observation']['sat_unit'] = PandExoParameters[rowCount][2] 
	exo_dict['observation']['noise_floor'] = PandExoParameters[rowCount][7]

	result = jdi.run_pandexo(exo_dict, ['NIRSpec G395M'], save_file=True, output_file=f'{PName}.p')
	
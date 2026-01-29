# use Group 1 Spectrum Code/SpectrumGenerator.py to read the csv file PandExoParametrers and update the parameters below accordingly

import os
import warnings
warnings.filterwarnings('ignore')

import csv
import pandexo.engine.justdoit as jdi 
import pandexo.engine.justplotit as jpi 

import numpy as np



# This section reads the PandExoParameters.csv file and extracts parameters
with open("Group 1 JWST Code/PandExoParameters.csv", newline="") as PandExoParametersFile:
	reader = csv.reader(PandExoParametersFile)
	header = next(reader)
	data = []
	planetNames = []
	for row in reader:

        cleaned = {}

        for key, value in row.items():
            value = value.strip()

            if value == "":
                cleaned[key] = np.nan
            else:
                try:
                    cleaned[key] = float(value)
                except ValueError:
                    cleaned[key] = value.strip("'")  # keep strings clean

        PandExoParameters.append(cleaned)
        planetNames.append(cleaned["Planet Name"])
#		row = [item.strip() for item in row]
#		if len(row) == 0:
#			continue
#		name, *nums = row
#		conv = [float(item) if item != "" else np.nan for item in nums]
#		data.append(conv)
#		planetNames.append(name)
#	PandExoParameters = np.array(data, dtype=float)



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
	exo_dict['planet']['exopath'] = f'Group 1 Spectrum Code/SpectrumPlots/TransmissionSpectrumTest{PName}.csv'              ## path to user defined spectrum file
	exo_dict['planet']['f_unit'] = PandExoParameters[rowCount][21]     ## flux unit for user defined spectrum
	exo_dict['planet']['w_unit'] = 'um'             ## wavelength unit for user defined spectra

	exo_dict['observation']['baseline'] = PandExoParameters[rowCount][5]
	exo_dict['observation']['baseline_unit'] = PandExoParameters[rowCount][6]
	exo_dict['observation']['noccultations'] = PandExoParameters[rowCount][3]  ## number of transits 
	exo_dict['observation']['sat_level'] = PandExoParameters[rowCount][1]            #saturation level in percent of full well 
	exo_dict['observation']['sat_unit'] = PandExoParameters[rowCount][2] 
	exo_dict['observation']['noise_floor'] = PandExoParameters[rowCount][7]

	result = jdi.run_pandexo(exo_dict, ['NIRSpec G395M'], save_file=True, output_file=f'JWSTSpectrum{PName}.csv')
	jpi.jwst_1d_spec(result, model=False, title=f'JWST Plot of {PName}', x_range=[1.6, 5.0])
	
'''
	clean_row = {
		k.strip(): v.strip() if v is not None else "" 
		for k, v in row.items()
		}
	planet_parameters.append(clean_row)

	# Helper function to convert strings to floats, handling empty strings
	def to_float(x):
		return float(x) if x != "" else np.nan

	# Example of accessing parameters for each planet
	for p in planet_parameters:
		planet_name = p["Planet Name"]

		try:
			Rp = (
				to_float(p["Planet Radius"]) * 
				radius_units[p["Planet R_unit"]]
				)
		except KeyError:
			raise ValueError(
				f'{planet_name}: Unrecognized planet radius unit "{p["Planet R_unit"]}"'
			)
		Rp_m = Rp.to(u.m).value

		try:
			Td = (
				to_float(p["Planet Transit_duration"]) 
				* time_units[p["Planet Td_unit"]]
				)
		except KeyError:
			raise ValueError(
				f'{planet_name}: Unrecognized transit duration unit "{p["Planet Td_unit"]}"'
			)
		Td_s = Td.to(u.s).value

	# TESTING BELOW TO VERIFY CORRECT READING OF PARAMETERS

	# Test 1: Correct number of planets
	print("Number of planets read:", len(planet_parameters))

	# Test 2: Print planet names
	print("\nPlanet names:")
	for p in planet_parameters:
		print(" ", p["Planet Name"])

	# Test 3: Spot-check numeric parsing
	print("\nNumeric sanity checks:")
	for p in planet_parameters:
		star_temp = float(p["Star Temp"])
		planet_radius = float(p["Planet Radius"])
		print(f" {p['Planet Name']}: T*={star_temp} K, Rp={planet_radius}")

	# Test 4: Check units survived
	print("\nUnit checks:")
	for p in planet_parameters:
		print(
			f" {p['Planet Name']}: "
			f"Rp unit = {p['Planet R_unit']}, "
			f"Transit unit = {p['Planet Td_unit']}"
		)
'''

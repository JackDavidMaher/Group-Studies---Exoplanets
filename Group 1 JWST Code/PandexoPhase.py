# use Group 1 Spectrum Code/SpectrumGenerator.py to read the csv file PandExoParametrers and update the parameters below accordingly

import csv
import os

# This section reads the PandExoParameters.csv file and extracts planetary parameters
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
	planetaryParameters = np.array(data, dtype=float)

rowCount=0
while rowCount < len(planetaryParameters):
	Rp = planetaryParameters[rowCount][0]  # Planet radius in units of Earth radii
	Mp = planetaryParameters[rowCount][1]  # Planet mass in units of Earth masses
	Tp = planetaryParameters[rowCount][2]  # Planet temperature in K
	mu = planetaryParameters[rowCount][3]  # Mean molecular weight in atomic mass units
	Pcloud = planetaryParameters[rowCount][4]  # Pressure at top of cloud deck in bar
	Pref = planetaryParameters[rowCount][5]   # Reference pressure in bar
	Rs = planetaryParameters[rowCount][6]  # Stellar radius in units of Solar radii     
	PName = planetNames[rowCount]  # Planet name (string)   
	Rp *= const.R_earth.value   # Convert Rp from units of R_Earth to m
	Rs *= const.R_sun.value     # Convert Rs from units of R_Sun to m
	Pcloud *= 1.0e5             # Convert Pcloud from bar to Pa
	Pref *= 1.0e5               # Convert Pref from bar to Pa
	mu *= sc.u                  # Convert mu from atomic mass units to kg
	#print(f"Processing planet with {header[0]}={Rp}, {header[1]}={Mp}, {header[2]}={Tp}, {header[3]}={mu}, {header[4]}={Pcloud}, {header[5]}={Pref}, {header[6]}={Rs}")
	#print(rowCount,len(planetaryParameters))
	rowCount=rowCount+1



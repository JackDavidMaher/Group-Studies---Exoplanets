import numpy as np
import scipy.constants as sc
from scipy.interpolate import RegularGridInterpolator
import astropy.constants as const
import matplotlib.pyplot as plt
import csv

with open("Group 1 Spectrum Code/PlanetaryParameters.csv", newline="") as PlanetaryParametersFile:
	reader = csv.reader(PlanetaryParametersFile)
	header = next(reader)
	data = []
	for row in reader:
		conv = [float(item.strip()) if item.strip() != "" else np.nan for item in row]
		data.append(conv)
planetaryParameters = np.array(data, dtype=float)

i=0
while i < len(planetaryParameters):
	Rp = planetaryParameters[i][0]  # Planet radius in units of Earth radii
	Mp = planetaryParameters[i][1]  # Planet mass in units of Earth masses
	Tp = planetaryParameters[i][2]  # Planet temperature in K
	mu = planetaryParameters[i][3]  # Mean molecular weight in atomic mass units
	Pcloud = planetaryParameters[i][4]  # Pressure at top of cloud deck in bar
	Pref = planetaryParameters[i][5]   # Reference pressure in bar
	Rs = planetaryParameters[i][6]  # Stellar radius in units of Solar radii        
	Rp *= const.R_earth.value   # Convert Rp from units of R_Earth to m
	Rs *= const.R_sun.value     # Convert Rs from units of R_Sun to m
	Pcloud *= 1.0e5             # Convert Pcloud from bar to Pa
	Pref *= 1.0e5               # Convert Pref from bar to Pa
	mu *= sc.u                  # Convert mu from atomic mass units to kg
	print(f"Processing planet with {header[0]}={Rp}, {header[1]}={Mp}, {header[2]}={Tp}, {header[3]}={mu}, {header[4]}={Pcloud}, {header[5]}={Pref}, {header[6]}={Rs}")
	i=i+1
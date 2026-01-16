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

print("Header:", header)
print("Planetary parameters array:\n", planetaryParameters)


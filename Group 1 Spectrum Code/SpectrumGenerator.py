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

	P = np.logspace(2.0,-5,100) * 1.0e5
	T = Tp*np.ones_like(P)
	n = P/(sc.k*T)
	rho = mu*n
	gp = (sc.G * Mp * const.M_earth.value) / (Rp)**2
	r = np.zeros_like(P)
	g = np.zeros_like(P)
	i_Rp = np.argmin(np.abs(P-Pref))
	r[i_Rp] = Rp
	g[i_Rp] = gp

	for i in range(i_Rp + 1, len(P)):
	g[i] = g[i_Rp] * r[i_Rp] * r[i_Rp] / (r[i-1] * r[i-1])
	r[i] = r[i-1] - ( sc.k * 0.5 * (T[i-1]+T[i]) / (mu * g[i]) ) * np.log(P[i]/P[i-1]) 

	for i in range(i_Rp-1, -1, -1):
	g[i] = g[i_Rp] * r[i_Rp] * r[i_Rp] / (r[i+1] * r[i+1])
	r[i] = r[i+1] - ( sc.k * 0.5 * (T[i+1]+T[i]) / (mu * g[i]) ) * np.log(P[i]/P[i+1])
	# First, set up a dictionary which will contain all the log mixing ratios, and input the abundances of all molecules except H2 and He
	logX = dict()
	logX['h2o'] = -3.0
	#logX['ch4'] = -3.0
	#logX['co'] = -3.0
	#logX['co2'] = -3.0
	#logX['nh3'] = -3.0

	# We'll also need to know the mean molecular weights of each molecule (here in units of amu)
	mmw = dict()
	mmw['h2o'] = 18.0
	mmw['h2'] = 2.0
	mmw['he'] = 4.0

	# Now we can 'fill' the rest of the atmosphere with H2 and He, which are typically the most abudnant background gases.
	# Let's assume that the ratio of H2 to He is the same as that in the Sun, i.e. X_He/X_H2 = 0.17 (Asplund et al. 2009 shows that X_He/X_H = 0.085, and going from H to H2 means doubling that)
	# Then X_H2 + X_He = 1 - X_rest

	X_rest = np.sum([np.power(10.0,logX[key]) for key in logX.keys()])
	X_H2 = (1.0 - X_rest) / (1.0 + 0.17)
	X_He = 0.17 * X_H2

	# Now let's add H2 and He to the logX dictionary

	logX['h2'] = np.log10(X_H2)
	logX['he'] = np.log10(X_He)

	# With all the mixing ratios defined, we can calculate the mean molecular weight of the atmosphere, mu:
	mu = 0.0
	for mol in logX.keys():
	mu += np.power(10.0,logX[mol])*mmw[mol]

	print("The mean molecular weight is:", mu)

import os
import numpy as np
import scipy.constants as sc
from scipy.interpolate import RegularGridInterpolator
import astropy.constants as const
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings('ignore')
import pandexo.engine.justdoit as jdi # THIS IS THE HOLY GRAIL OF PANDEXO
import numpy as np
import pandexo.engine.justplotit as jpi 
import pandas as pd


## CHANGE PATH IF NEED BE ##
filedirectory = '165 planets data'   ##  name of folder just change number

with open("/Code/All_165_valid_planets_under_1000K05.02_16-10.csv", newline="") as PlanetaryParametersFile:   
	reader = csv.reader(PlanetaryParametersFile)
	header = next(reader)
	data = []
	planetNames = []
	for row in reader:
		row = [item.strip() for item in row]
		if len(row) == 0:
			continue
		name, *nums = row
		conv = [float(item) if item != "" else np.nan for item in nums]
		data.append(conv)
		planetNames.append(name)
	planetaryParameters = np.array(data, dtype=float)
	
rowCount= 0 
while rowCount < len(planetaryParameters):
	PName = planetNames[rowCount]                                   ## planet name   
	print(f"---Analysing Planet: {PName} ----")
	print(f"Star Magnitude: {planetaryParameters[rowCount][20]}")

    ## CHANGE INDEXES IF NEED BE ##
	
	Rp = planetaryParameters[rowCount][3]                           ## planet radius in units of Earth radii
	Mp = planetaryParameters[rowCount][2]                           ## planet mass in units of Earth masses
	Tp = planetaryParameters[rowCount][39]                          ## planet temperature in K
	mu = 4.88                                                       ## mean molecular weight in atomic mass units
	Pcloud = 100 
	Pref = 0.01                                                     ## pressure at top of cloud deck in bar  
	Rs = planetaryParameters[rowCount][14]                          ## stellar radius in units of Solar radii     
	Rp *= const.R_earth.value                                       ## Convert Rp from units of R_Earth to m
	Rs *= const.R_sun.value                                         ## Convert Rs from units of R_Sun to m
	Pcloud *= 1.0e5                                                 ## Convert Pcloud from bar to Pa
	Pref *= 1.0e5                                                   ## Convert Pref from bar to Pa
	mu *= sc.u                                                      ## Convert mu from atomic mass units to kg

	

	P = np.logspace(2.0,-9,100) * 1.0e5
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
	logX['h2o'] = -1.1        ## fixed compositions
	logX['ch4'] = -1.74
	logX['co'] = -2.0
	logX['co2'] = -1.7
	#logX['nh3'] = -3.0

	## We'll also need to know the mean molecular weights of each molecule (here in units of amu) ##
	mmw = dict()
	mmw['h2o'] = 18.0
	mmw['ch4'] = 16.0
	mmw['co'] = 28.0
	mmw['co2'] = 44.0
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

	#print("The mean molecular weight is:", mu)

	xsec_dict = dict()
	lam_dict = dict()
	P_dict = dict()
	T_dict = dict()

	for mol in mmw.keys():
		if mol =='h2':
			break
		else:
			xsec_dict[mol] = np.load(f'/Group 1 Full Loop Code/GivenResources/cross_section_files/Cross_section_files/{mol}_xsec.npy') #cross-section
			lam_dict[mol] = np.load(f'/Group 1 Full Loop Code/GivenResources/cross_section_files/Cross_section_files/{mol}_lam.npy')*1e6 # convert to microns, wavelength
			P_dict[mol] = np.power(10.0,np.load(f'/Group 1 Full Loop Code/GivenResources/cross_section_files/Cross_section_files/{mol}_P.npy')) # already in Pa, pressure
			T_dict[mol] = np.load(f'/Group 1 Full Loop Code/GivenResources/cross_section_files/Cross_section_files/{mol}_T.npy') # temperature


	# H2-H2 and He-H2 molecule pairs cause absorption through a process called "collision-induced absorption". This data is wavelength- and temperature-dependent, but not pressure-dependent.
	xsec_h2h2 = np.load('/Group 1 Full Loop Code/GivenResources/cross_section_files/Cross_section_files/h2_h2_xsec.npy')
	lam_h2h2 = np.load('/Group 1 Full Loop Code/GivenResources/cross_section_files/Cross_section_files/h2_h2_lam.npy')

	xsec_heh2 = np.load('/Group 1 Full Loop Code/GivenResources/cross_section_files/Cross_section_files/he_h2_xsec.npy')
	lam_heh2 = np.load('/Group 1 Full Loop Code/GivenResources/cross_section_files/Cross_section_files/he_h2_lam.npy')

	lam = np.linspace(0.61,5.0,200)

	log_xsec_dict = dict()

    #all molecules
	for mol in logX.keys():
		if mol =='h2':
			break
		interp_xsec = RegularGridInterpolator((lam_dict[mol], P_dict[mol], T_dict[mol]), xsec_dict[mol], method='linear', bounds_error=False, fill_value=None)
		lamlam, PP = np.meshgrid(lam, P, indexing="ij")
    	# Ensure requested temperature is inside the interpolator grid (avoid out-of-bounds)
		T0 = np.clip(T[0], T_dict[mol].min(), T_dict[mol].max())
		# Build points with shape (npoints, ndim) for the interpolator, then reshape back
		pts = np.vstack((lamlam.ravel(), PP.ravel(), np.full(lamlam.size, T0))).T
		log_xsec_dict[mol] = interp_xsec(pts).reshape(lamlam.shape)  # this assumes an isothermal atmosphere

	log_cia_dict = dict()
	log_cia_dict['h2h2'] = np.interp(lam, lam_h2h2, xsec_h2h2)
	log_cia_dict['heh2'] = np.interp(lam, lam_heh2, xsec_heh2)

	sum_nsigma = np.zeros((len(lam), len(P)))
	for mol in log_xsec_dict.keys():
		sum_nsigma += n[np.newaxis,:]*np.power(10.0,logX[mol])*np.power(10.0,log_xsec_dict[mol])

	sum_nsigma += n[np.newaxis,:]*n[np.newaxis,:]*np.power(10.0,logX['h2'])*np.power(10.0,logX['h2'])*np.power(10.0,log_cia_dict['h2h2'])[:,np.newaxis]
	sum_nsigma += n[np.newaxis,:]*n[np.newaxis,:]*np.power(10.0,logX['he'])*np.power(10.0,logX['h2'])*np.power(10.0,log_cia_dict['heh2'])[:,np.newaxis]

	integral_gt_Rp = np.zeros((len(lam))) # initialise an array where we'll store the "r>Rp" integral
	integral_lt_Rp = np.zeros((len(lam))) # initialise an array where we'll store the "r<Rp" integral

	exptau = np.zeros((len(P),len(lam)))
	transit_depth = np.zeros((len(lam)))

	# Compute the total optical depth at each impact parameter and wavelength #

	opacity = sum_nsigma #(kappa+sigma)
	# For each layer of atmosphere

	for i in range(len(r)-1):

		s_tot = np.sqrt(r[i:]*r[i:]-r[i]*r[i])
		ds = s_tot[1:]-s_tot[:-1]

		tau_tot = np.sum((opacity[:, i:-1] + opacity[:, i+1:])*ds[np.newaxis,:],axis=-1)

		# If r[i] is deeper that the top of the cloud, the atmosphere here is fully opaque and tau is very large.
		if P[i] > Pcloud:
			tau_tot += 1000.0

		# We need e^(-tau) for the integral, so let's calculate that here. Adding 1.0-e-250 avoids errors when taking the log of this in cases where e^-tau is essentially zero.
		exptau[i, :] = np.exp(-1.0*tau_tot) + 1.0e-250


	# Compute transit depth terms. Note that we take an average of the i^th and (i+1)^th terms in the integrals.
	for i in range(len(r)-1):

		# Rays travelling through atmosphere above Rp
		if (r[i] >= Rp):
			integral_gt_Rp[:] += 0.5*((r[i]*(1.0 - exptau[i, :]) + (r[i+1]*(1.0 - exptau[i+1, :])))*(r[i+1] - r[i]))

		# Rays travelling through atmosphere below Rp
		if (r[i] < Rp):
			integral_lt_Rp[:] += 0.5*((r[i]*(exptau[i, :]) + (r[i+1]*(exptau[i+1, :])))*(r[i+1] - r[i]))

	# Compute effective transit depth (transmission spectrum) #
	transit_depth[:] = (Rp*Rp + 2.0*integral_gt_Rp[:] - 2.0*integral_lt_Rp[:])/(Rs*Rs)

	plt.Figure(figsize=(12,8))
	plt.plot(lam,transit_depth*1e6) #convert transit_depth into units of ppm
	plt.xlabel('Wavelength (microns)')
	plt.ylabel('Transit Depth (ppm)')
	plt.title(f'Transmission Spectrum of {PName}')
	plt.xlim([2.5,5.0])


	plt.savefig(f'/Group 1 Full Loop Code/{filedirectory}/spectrum plots/planet_spectrum_{PName}.png')
	plt.close()
	np.savetxt(f'/Group 1 Full Loop Code/{filedirectory}/spectrum txt files/planet_spectrum_{PName}.txt', np.column_stack((lam, transit_depth)), header='Wavelength(micron)   Transit_Depth(rp^2/r*^2)', fmt='%10.6f')


	exo_dict = jdi.load_exo_dict()
    ## star dict
	exo_dict['star']['type'] = 'phoenix'     
	exo_dict['star']['temp'] = planetaryParameters[rowCount][16]                 ## temperature in K 
	exo_dict['star']['metal'] = planetaryParameters[rowCount][18]                ## metallacity as log Fe/H
	exo_dict['star']['logg'] = planetaryParameters[rowCount][22]                 ## log gravity cgs
	exo_dict['star']['mag'] = planetaryParameters[rowCount][20]                  ## star J magnitude
	exo_dict['star']['ref_wave'] = 1.25
	exo_dict['star']['radius'] = planetaryParameters[rowCount][14]               ##radius of the star in solar radii
	exo_dict['star']['r_unit'] = 'R_sun'

    ## planet dict
	exo_dict['planet']['radius'] = planetaryParameters[rowCount][3]              ##radius of the planet in earth radii        
	exo_dict['planet']['r_unit'] = 'R_earth'                                     ## or R_jup
	exo_dict['planet']['transit_duration'] = planetaryParameters[rowCount][8]    ##transit duration in days
	exo_dict['planet']['td_unit'] = 'h'
	exo_dict['planet']['type'] = 'user'                                          ## 'user' for user defined spectrum or 'constant' for constant spectrum
	exo_dict['planet']['exopath'] = f'/Group 1 Full Loop Code/165 planets data/spectrum txt files/planet_spectrum_{PName}.txt'       ## path to user defined spectrum file
	exo_dict['planet']['f_unit'] = 'rp^2/r*^2'                                   ## flux unit for user defined spectrum
	exo_dict['planet']['w_unit'] = 'um'                                          ## wavelength unit for user defined spectra

	exo_dict['observation']['baseline'] = 2.0 
	exo_dict['observation']['baseline_unit'] = 'frac'
	exo_dict['observation']['noccultations'] = 1                                 ## number of transits (changed to match num_tran=10 in plot)
	exo_dict['observation']['sat_level'] = 80                                    ## saturation level in percent of full well 
	exo_dict['observation']['sat_unit'] = '%' 
	exo_dict['observation']['noise_floor'] = 0

	result = jdi.run_pandexo(exo_dict, ['NIRSpec G395H'] ,save_file=False)

	spec_dict = jpi.jwst_1d_spec(result, R=500, model=True, title=f'JWST Plot of {PName}', x_range=[2.8, 5.0], plot=False)

	wavelength = result['FinalSpectrum']['wave']
	observed_depth = result['FinalSpectrum']['spectrum_w_rand'] # Data + Noise
	model_depth = result['FinalSpectrum']['spectrum']          # The smooth model
	errors = result['FinalSpectrum']['error_w_floor']   # The 1-sigma uncertainties
	
	plt.errorbar(wavelength, model_depth, yerr=errors, fmt='s', color='royalblue', 
             markersize=1, alpha=0.1, label=f'{PName} Simulated Data')      
	plt.ylim([min(model_depth)*0.9, max(model_depth)*1.1])
	plt.xlabel('Wavelength ($\mu$m)', fontsize=8)
	plt.ylabel('Transit Depth (ppm)', fontsize=8)
	plt.title(f'PandExo Simulated Observation for {PName}', fontsize=10)
	plt.xlim(2.8,5)
	plt.legend(frameon=True)
	plt.grid(True, alpha=0.3)
	plt.savefig(f'/Group 1 Full Loop Code/{filedirectory}/JWST plots/{PName}_JWST_simulated_observation.png')
	plt.close()
	df = pd.DataFrame({
    'Wavelength_um': wavelength,
    'Transit_Depth_ppm': observed_depth * 1e6,
    'Error_ppm': errors * 1e6 })

	df.to_csv(f'/Group 1 Full Loop Code/{filedirectory}/pandexo csv files/{PName}_JWST_results.csv', index=False)

	print(f'---Finished analysing planet: {PName}---')
	rowCount += 1
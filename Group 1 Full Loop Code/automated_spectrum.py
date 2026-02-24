## --------------------- LOADING IN ALL NECESSARY PACKAGES AND DIRECTORIES --------------------- ##
import os, csv, warnings 
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import scipy.constants as sc
import matplotlib.pyplot as plt
import astropy.constants as const
from scipy.interpolate import RegularGridInterpolator
from dotenv import load_dotenv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
# This sets the environment variables for pandeia_refdata and PYSYN_CDBS to the values in the .env file, which should be set to the correct paths on your system. If these environment variables are already set in your system, this will not change them.

## CHANGE PATH IF NEED BE ##
filedirectory = '20 planets data'   ##  name of folder just change number

xsec_h2o = np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/h2o_xsec.npy')
lam_h2o = np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/h2o_lam.npy') * 1e6 # convert to microns
P_h2o = np.power(10.0,np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/h2o_P.npy')) # already in Pa
T_h2o = np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/h2o_T.npy')

# Read in CO2 cross section data
xsec_co2 = np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/co2_xsec.npy')
lam_co2 = np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/co2_lam.npy') * 1e6 # convert to microns
P_co2 = np.power(10.0,np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/co2_P.npy')) # already in Pa
T_co2 = np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/co2_T.npy')

# Read in CH4 cross section data
xsec_ch4 = np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/ch4_xsec.npy')
lam_ch4 = np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/ch4_lam.npy') * 1e6 # convert to microns
P_ch4 = np.power(10.0,np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/ch4_P.npy')) # already in Pa
T_ch4 = np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/ch4_T.npy')

#read in CO cross section data
xsec_co = np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/co_xsec.npy')
lam_co = np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/co_lam.npy') * 1e6 # convert to microns   
P_co = np.power(10.0,np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/co_P.npy')) # already in Pa
T_co = np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/co_T.npy')

# H2-H2 and He-H2 molecule pairs cause absorption through a process called "collision-induced absorption". This data is wavelength- and temperature-dependent, but not pressure-dependent.
xsec_h2h2 = np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/h2_h2_xsec.npy')
lam_h2h2 = np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/h2_h2_lam.npy')

xsec_heh2 = np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/he_h2_xsec.npy')
lam_heh2 = np.load(f'{PROJECT_DIR}/GivenResources/cross_section_files/Cross_section_files/he_h2_lam.npy')

import pandexo.engine.justdoit as jdi 

with open(f'{PROJECT_DIR}/Code/Data/20_top_f_h_planets17.02_16-44.csv', newline="") as planetaryparametersfile:   ## Add name of file
	reader = csv.reader(planetaryparametersfile)
	header = next(reader)
	
	data = []
	planetnames = []
	
	for row in reader:
		row = [item.strip() for item in row]
		if len(row) == 0:
			continue
		name, *nums = row
		conv = [float(item) if item != "" else np.nan for item in nums]
		data.append(conv)
		planetnames.append(name)
	
	planetaryparameters = np.array(data, dtype=float)

def bin_spectrum(wave, flux, model, error, bin_width = 0.0225):
    bins = np.arange(min(wave), max(wave), bin_width)
    binned_wave = []
    binned_flux = []
    binned_model = []
    binned_err = []
    for i in range(len(bins) - 1):
        indices = np.where((wave >= bins[i]) & (wave < bins[i+1]))[0]
        if len(indices) > 0:
            w = wave[indices]
            f = flux[indices]
            m = model[indices] 
            e = error[indices]

            weights = 1.0 / (e**2)
            weighted_f = np.sum(f * weights) / np.sum(weights)
            weighted_m = np.sum(m * weights) / np.sum(weights)
            propagated_e = np.sqrt(1.0 / np.sum(weights))
            
            binned_wave.append(np.mean(w))
            binned_flux.append(weighted_f)
            binned_model.append(weighted_m)
            binned_err.append(propagated_e)
            
    return np.array(binned_wave), np.array(binned_flux),np.array(binned_model), np.array(binned_err)

def pressure(t):
    if t >= 600:
        p = 0.01 * t - 10
    elif t < 600:
        p = -0.01 * t + 2
    return p

rowcount = 0 
while rowcount < len(planetaryparameters):
	## ----------------------------- LOAD PLANETARY PARAMETERS ------------------------------ ##
	planet_id = planetnames[rowcount]                               ## planet name   
	print(f"------------ Analysing Planet: {planet_id} --------------")
	
	Rp = planetaryparameters[rowcount][3] * const.R_earth.value     ## planet radius in units of Earth radii * Earth radius in m
	Mp = planetaryparameters[rowcount][2]                           ## planet mass in units of Earth masses
	Tp = planetaryparameters[rowcount][39]                          ## planet temperature in K
	Rs = planetaryparameters[rowcount][14] * const.R_sun.value      ## stellar radius in units of Solar radii * Solar radius in m    
	
	mu = 4.88 * sc.u                                       		   ## mean molecular weight in atomic mass units * atomic mass unit in kg
	Pcloud = 10 ** pressure(Tp) * 1.0e5                                  		   ## pressure at top of cloud deck in Pa (100 bar) * convert to Pa
	Pref = 0.01 * 1.0e5                                   		   ## pressure at top of cloud deck in bar * convert to Pa 

	## --------------------- ATOMOSPHERIC COMPOSOTION/ABSOBTION SPECTRA --------------------- ##

	P = np.logspace(2.0, -9.0, 500) * 1.0e5
	T = Tp * np.ones_like(P)
	n = P / (sc.k * T)
	rho = mu * n
	gp = (sc.G * Mp * const.M_earth.value) / (Rp)**2
	r = np.zeros_like(P)
	g = np.zeros_like(P)
	i_Rp = np.argmin(np.abs(P-Pref))
	r[i_Rp] = Rp
	g[i_Rp] = gp

	# First, set up a dictionary which will contain all the log mixing ratios, and input the abundances of all molecules except H2 and He
	logX = dict()
	logX['h2o'] = -1.1
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

	for i in range(i_Rp + 1, len(P)):
		g[i] = g[i_Rp] * r[i_Rp] * r[i_Rp] / (r[i-1] * r[i-1])
		r[i] = r[i-1] - ( sc.k * 0.5 * (T[i-1]+T[i]) / (mu * g[i]) ) * np.log(P[i]/P[i-1]) 

	for i in range(i_Rp-1, -1, -1):
		g[i] = g[i_Rp] * r[i_Rp] * r[i_Rp] / (r[i+1] * r[i+1])
		r[i] = r[i+1] - ( sc.k * 0.5 * (T[i+1]+T[i]) / (mu * g[i]) ) * np.log(P[i]/P[i+1])

	# With all the mixing ratios defined, we can calculate the mean molecular weight of the atmosphere, mu:
	mu = 0.0
	for mol in logX.keys():
		mu += np.power(10.0,logX[mol])*mmw[mol]

	xsec_dict = dict()
	lam_dict = dict()
	P_dict = dict()
	T_dict = dict()
	

	lam = np.linspace(0.61,5.0,5000)

	log_xsec_dict = dict()

	lamlam, PP = np.meshgrid(lam, P, indexing="ij")

	# ---- H2O ----
	interp_h2o = RegularGridInterpolator(
		(lam_h2o, P_h2o, T_h2o),
		xsec_h2o,
		method='linear', bounds_error=False,
		fill_value=None)
	log_xsec_dict['h2o'] = interp_h2o((lamlam, PP, T[0]))

	# ---- CO2 ----
	interp_co2 = RegularGridInterpolator(
		(lam_co2, P_co2, T_co2),
		xsec_co2,
		method='linear', bounds_error=False,
		fill_value=None
	)
	log_xsec_dict['co2'] = interp_co2((lamlam, PP, T[0]))

	# ---- CH4 ----
	interp_ch4 = RegularGridInterpolator(
		(lam_ch4, P_ch4, T_ch4),
		xsec_ch4,
		method='linear', bounds_error=False,
		fill_value=None
	)
	log_xsec_dict['ch4'] = interp_ch4((lamlam, PP, T[0]))

	interp_co2 = RegularGridInterpolator(
		(lam_co, P_co, T_co),
		xsec_co,
		method='linear', bounds_error=False,
		fill_value=None
	)
	log_xsec_dict['co'] = interp_co2((lamlam, PP, T[0]))

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

	plt.figure(figsize=(12,8))
	plt.plot(lam,transit_depth * 1e6) #convert transit depth into units of ppm
	plt.xlabel('Wavelength (um)')
	plt.ylabel('Transit Depth (ppm)')
	plt.title(f'[{planet_id}] Transmission Spectrum')
	plt.xlim([2.8,5.0])
	plt.savefig(f'{PROJECT_DIR}/Group 1 Full Loop Code/{filedirectory}/spectrum plots/planet_spectrum_{planet_id}.png')
	plt.close()

	np.savetxt(f'{PROJECT_DIR}/Group 1 Full Loop Code/{filedirectory}/spectrum txt files/planet_spectrum_{planet_id}.txt', np.column_stack((lam, transit_depth)), header='Wavelength(micron)   Transit_Depth(rp^2/r*^2)', fmt='%10.6f')

	## ------------------------------------- SCALE HEIGHT --------------------------------- ##

	left_mask = (lam >= 4.1) & (lam <= 4.15)
	
	left_depth_values = transit_depth[left_mask]
	left_lam_values = lam[left_mask]
	
	left_val = np.mean(left_depth_values)
	left_lam_target = np.mean(left_lam_values)
	peak_mask = (lam >= 4.25) & (lam <= 4.28) 

	peak_region_depths = transit_depth[peak_mask]
	peak_x_values = lam[peak_mask]

	peak_val = np.mean(peak_region_depths)
	peak_x_value = np.mean(peak_x_values)
	feature_height_ratio = (peak_val - left_val)
	feature_height_ppm = (peak_val - left_val) * 1e6

	peak_scatter_error_ppm = np.std(peak_region_depths) * 1e6
	scale_height = (sc.k * Tp) / (mu * gp * sc.m_p)

	A_H = feature_height_ratio * (Rs ** 2) / (2 * scale_height * Rp)
	print(f'Estimated scale height for {planet_id}: {A_H:.2f} m')

	## ----------------------------------- PANDEXO PLOTTING ----------------------------------- ##

	exo_dict = jdi.load_exo_dict()
    
	## star dictionary
	exo_dict['star']['type'] = 'phoenix'     
	exo_dict['star']['temp'] = planetaryparameters[rowcount][16]                 ## temperature in K 
	exo_dict['star']['metal'] = planetaryparameters[rowcount][18]                ## metallacity as log Fe/H
	exo_dict['star']['logg'] = planetaryparameters[rowcount][22]                 ## log gravity cgs
	exo_dict['star']['mag'] = planetaryparameters[rowcount][20]                  ## star J magnitude
	exo_dict['star']['ref_wave'] = 1.25
	exo_dict['star']['radius'] = planetaryparameters[rowcount][14]               ## radius of the star in solar radii
	exo_dict['star']['r_unit'] = 'R_sun'

    ## planet dictionary
	exo_dict['planet']['radius'] = planetaryparameters[rowcount][3]              ## radius of the planet in earth radii        
	exo_dict['planet']['r_unit'] = 'R_earth'                                    
	exo_dict['planet']['transit_duration'] = planetaryparameters[rowcount][8]    ## transit duration in days
	exo_dict['planet']['td_unit'] = 'h'
	exo_dict['planet']['type'] = 'user'                                          ## 'user' for user defined spectrum or 'constant' for constant spectrum
	exo_dict['planet']['exopath'] = f'Group 1 Full Loop Code/{filedirectory}/spectrum txt files/planet_spectrum_{planet_id}.txt'       ## path to user defined spectrum file
	exo_dict['planet']['f_unit'] = 'rp^2/r*^2'                                   ## flux unit for user defined spectrum
	exo_dict['planet']['w_unit'] = 'um'                                          ## wavelength unit for user defined spectra
	
	## Error and observation parameters
	exo_dict['observation']['baseline'] = 2.0 
	exo_dict['observation']['baseline_unit'] = 'frac'
	exo_dict['observation']['noccultations'] = 1                                 ## number of transits (changed to match num_tran=10 in plot)
	exo_dict['observation']['sat_level'] = 80                                    ## saturation level in percent of full well 
	exo_dict['observation']['sat_unit'] = '%' 
	exo_dict['observation']['noise_floor'] = 0

	result = jdi.run_pandexo(exo_dict, ['NIRSpec G395H'], save_file = False, verbose = False)

	wavelength = result['FinalSpectrum']['wave']
	observed_depth = result['FinalSpectrum']['spectrum_w_rand'] # Data + Noise
	model_depth = result['FinalSpectrum']['spectrum']          # The smooth model
	errors = result['FinalSpectrum']['error_w_floor']   # The 1-sigma uncertainties
	
	plt.errorbar(wavelength, observed_depth, yerr=errors, fmt='s', color='royalblue', markersize=1, alpha=0.1, label=f'{planet_id} Simulated Data', zorder = 1)
	plt.plot(wavelength, model_depth, color = 'firebrick', zorder = 2)      
	plt.ylim([min(model_depth) * 0.9, max(model_depth) * 1.1])
	plt.xlabel('Wavelength ($\mu$m)', fontsize=8)
	plt.ylabel('Transit Depth (ppm)', fontsize=8)
	plt.title(f'PandExo Simulated Observation for {planet_id}', fontsize=10)
	plt.xlim(2.8,5)
	plt.legend(frameon=True)
	plt.grid(True, alpha=0.3)
	plt.savefig(f'{PROJECT_DIR}/Group 1 Full Loop Code/{filedirectory}/JWST plots/{planet_id}_JWST_simulated_observation.png')
	plt.close()

	df = pd.DataFrame({
    'Wavelength_um': wavelength,
	'Model_Depth': model_depth,
    'Transit_Depth': observed_depth,
    'Error': errors})
	df.to_csv(f'{PROJECT_DIR}/Group 1 Full Loop Code/{filedirectory}/pandexo csv files/{planet_id}_JWST_results.csv', index=False)

	print(f'-------------- Finished analysing planet: {planet_id} ----------')
	rowcount += 1
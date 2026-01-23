import numpy as np
import scipy.constants as sc
from scipy.interpolate import RegularGridInterpolator
import astropy.constants as const
import matplotlib.pyplot as plt
import csv
<<<<<<< HEAD
import os
import subprocess
import shutil
import sys

# This section runs the SpectrumImageCleaner to remove any existing spectrums from the computer so dupliucates are not created
script_dir = os.path.dirname(os.path.abspath(__file__))
c_src = os.path.join(script_dir, 'SpectrumImageCleaner.c')
bin_path = os.path.join(script_dir, 'SpectrumImageCleaner')
if not os.path.isfile(bin_path) or not os.access(bin_path, os.X_OK):
	if not os.path.isfile(c_src):
		sys.exit(f"SpectrumImageCleaner binary not found and source {c_src} missing.")
	gcc = shutil.which('gcc')
	if gcc is None:
		sys.exit("gcc not found; cannot compile SpectrumImageCleaner.c")
	compile_proc = subprocess.run([gcc, '-O2', '-o', bin_path, c_src], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	if compile_proc.returncode != 0:
		sys.exit(f"Failed to compile SpectrumImageCleaner.c:\n{compile_proc.stderr}")
run_proc = subprocess.run([bin_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=script_dir)
if run_proc.returncode != 0:
	sys.exit(f"SpectrumImageCleaner failed:\n{run_proc.stderr}")
else:
	if run_proc.stdout:
		print(run_proc.stdout)




=======
>>>>>>> 4d54eaf (Jack maher (#40))
import os
import subprocess
import shutil
import sys

# This section runs the SpectrumImageCleaner to remove any existing spectrums from the computer so dupliucates are not created
script_dir = os.path.dirname(os.path.abspath(__file__))
c_src = os.path.join(script_dir, 'SpectrumImageCleaner.c')
bin_path = os.path.join(script_dir, 'SpectrumImageCleaner')
if not os.path.isfile(bin_path) or not os.access(bin_path, os.X_OK):
	if not os.path.isfile(c_src):
		sys.exit(f"SpectrumImageCleaner binary not found and source {c_src} missing.")
	gcc = shutil.which('gcc')
	if gcc is None:
		sys.exit("gcc not found; cannot compile SpectrumImageCleaner.c")
	compile_proc = subprocess.run([gcc, '-O2', '-o', bin_path, c_src], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
	if compile_proc.returncode != 0:
		sys.exit(f"Failed to compile SpectrumImageCleaner.c:\n{compile_proc.stderr}")
run_proc = subprocess.run([bin_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=script_dir)
if run_proc.returncode != 0:
	sys.exit(f"SpectrumImageCleaner failed:\n{run_proc.stderr}")
else:
	if run_proc.stdout:
		print(run_proc.stdout)



with open("Group 1 Spectrum Code/PlanetaryParameters.csv", newline="") as PlanetaryParametersFile:
	reader = csv.reader(PlanetaryParametersFile)
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
	logX['h2o'] = 0
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

	#print("The mean molecular weight is:", mu)



	xsec_h2o = np.load('GivenResources/cross_section_files/Cross_section_files/h2o_xsec.npy')
	lam_h2o = np.load('GivenResources/cross_section_files/Cross_section_files/h2o_lam.npy')*1e6 # convert to microns
	P_h2o = np.power(10.0,np.load('GivenResources/cross_section_files/Cross_section_files/h2o_P.npy')) # already in Pa
	T_h2o = np.load('GivenResources/cross_section_files/Cross_section_files/h2o_T.npy')

	# H2-H2 and He-H2 molecule pairs cause absorption through a process called "collision-induced absorption". This data is wavelength- and temperature-dependent, but not pressure-dependent.
	xsec_h2h2 = np.load('GivenResources/cross_section_files/Cross_section_files/h2_h2_xsec.npy')
	lam_h2h2 = np.load('GivenResources/cross_section_files/Cross_section_files/h2_h2_lam.npy')

	xsec_heh2 = np.load('GivenResources/cross_section_files/Cross_section_files/he_h2_xsec.npy')
	lam_heh2 = np.load('GivenResources/cross_section_files/Cross_section_files/he_h2_lam.npy')

	lam = np.linspace(0.61,5.0,200)

	log_xsec_dict = dict()
	interp_xsec = RegularGridInterpolator((lam_h2o, P_h2o, T_h2o), xsec_h2o, method='linear', bounds_error=False, fill_value=None)
	lamlam, PP = np.meshgrid(lam, P, indexing="ij")

	# Ensure requested temperature is inside the interpolator grid (avoid out-of-bounds)
	T0 = np.clip(T[0], T_h2o.min(), T_h2o.max())
	# Build points with shape (npoints, ndim) for the interpolator, then reshape back
	pts = np.vstack((lamlam.ravel(), PP.ravel(), np.full(lamlam.size, T0))).T
	log_xsec_dict['h2o'] = interp_xsec(pts).reshape(lamlam.shape)  # this assumes an isothermal atmosphere

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


	plt.plot(lam,transit_depth*1e6) #convert transit_depth into units of ppm
	plt.xlabel('Wavelength (microns)')
	plt.ylabel('Transit Depth (ppm)')
	plt.title(f'Transmission Spectrum of {PName}')
	#plt.xlim([1.1,1.7])

	# Ensure output directory exists and save CSV for the plotted spectrum
	plots_dir = os.path.join(script_dir, 'SpectrumPlots')
	os.makedirs(plots_dir, exist_ok=True)
	safe_name = PName.replace(' ', '_').replace('/', '_')
	csv_path = os.path.join(plots_dir, f'TransmissionSpectrum_{safe_name}.csv')
	with open(csv_path, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(['wavelength_micron', 'transit_depth_ppm'])
		for w, td in zip(lam, transit_depth*1e6):
			writer.writerow([f"{w:.6e}", f"{td:.12e}"])

	# Save the plot into the same output directory
	plt.savefig(os.path.join(plots_dir, f'TransmissionSpectrum_{safe_name}.png'))
	plt.clf()
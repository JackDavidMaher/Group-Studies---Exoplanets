import numpy as np
from scipy.signal import savgol_filter

def measure_band_feature(
    lam_um, depth_ppm,
    band=(2.7, 3.7),          # methane ~3.3 µm band window (edit as needed)
    sidebands=((2.9, 3.1), (3.85, 3.95)),  # continuum windows either side
    smooth=True,
    sg_window=21, sg_poly=3
):
    lam = np.asarray(lam_um)
    y = np.asarray(depth_ppm)

    # Optional smoothing (keep window odd and < len(data))
    if smooth:
        w = min(sg_window, len(y) - (1 - len(y) % 2))  # ensure not too long
        if w % 2 == 0: w -= 1
        if w >= 5:
            y_s = savgol_filter(y, window_length=w, polyorder=min(sg_poly, w-2))
        else:
            y_s = y.copy()
    else:
        y_s = y.copy()

    # Masks for band + sidebands
    band_mask = (lam >= band[0]) & (lam <= band[1])
    sb_mask = ((lam >= sidebands[0][0]) & (lam <= sidebands[0][1])) | \
              ((lam >= sidebands[1][0]) & (lam <= sidebands[1][1]))

    if band_mask.sum() < 3 or sb_mask.sum() < 4:
        raise ValueError("Not enough points in band/sidebands; widen windows or check wavelength grid.")

    # Fit linear continuum: y_cont = a*lam + b using sidebands
    A = np.vstack([lam[sb_mask], np.ones(sb_mask.sum())]).T
    a, b = np.linalg.lstsq(A, y_s[sb_mask], rcond=None)[0]

    # Continuum across band
    cont_band = a * lam[band_mask] + b
    y_band = y_s[band_mask]
    lam_band = lam[band_mask]

    # Peak location in the band (max transit depth)
    i_peak = np.argmax(y_band)
    lam_peak = lam_band[i_peak]
    peak_ppm = y_band[i_peak]
    cont_at_peak = cont_band[i_peak]
    amp_peak_ppm = peak_ppm - cont_at_peak

    # Band-averaged amplitude (often more stable than a single max point)
    amp_mean_ppm = np.mean(y_band - cont_band)

    # Noise estimate from sidebands residuals
    resid_sb = y_s[sb_mask] - (a * lam[sb_mask] + b)
    sigma_ppm = np.std(resid_sb, ddof=2)

    # Rough uncertainties:
    # - peak amplitude uncertainty ~ sigma
    # - mean amplitude uncertainty ~ sigma/sqrt(Nband)
    Nband = band_mask.sum()
    amp_peak_err = sigma_ppm
    amp_mean_err = sigma_ppm / np.sqrt(Nband)

    return {
        "band": band,
        "sidebands": sidebands,
        "lam_peak_um": float(lam_peak),
        "amp_peak_ppm": float(amp_peak_ppm),
        "amp_peak_err_ppm": float(amp_peak_err),
        "amp_mean_ppm": float(amp_mean_ppm),
        "amp_mean_err_ppm": float(amp_mean_err),
        "continuum_slope_ppm_per_um": float(a),
        "continuum_intercept_ppm": float(b),
        "sigma_sidebands_ppm": float(sigma_ppm),
        "Nband": int(Nband),
    }

# ---- Example usage on your file ----
# File format: wavelength(um) depth(fraction)
data = np.loadtxt("/Users/thomasmorgan/Documents/Group-Studies---Exoplanets/Notebooks/TransmissionSpectrum_TestPlanetB.txt")  # :contentReference[oaicite:0]{index=0}
lam_um = data[:, 0]
depth_frac = data[:, 1]
depth_ppm = depth_frac * 1e6

# Measure the ~3.3 µm CH4 feature (adjust windows for your case)
result = measure_band_feature(
    lam_um, depth_ppm,
    band=(3.1, 3.7),
    sidebands=((2.9, 3.1), (3.85, 3.95)),
    smooth=True
)

print(result)

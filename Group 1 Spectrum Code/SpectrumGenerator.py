import numpy as np
import scipy.constants as sc
from scipy.interpolate import RegularGridInterpolator
import astropy.constants as const
import matplotlib.pyplot as plt
import csv
from pathlib import Path


def load_planetary_parameters(path: str | None = None):
	"""Load PlanetaryParameters.csv and return (header, numpy array).

	Empty fields are converted to np.nan.
	If `path` is None, the function looks for `PlanetaryParameters.csv`
	in the same directory as this script.
	"""
	p = Path(path) if path else (Path(__file__).resolve().parent / "PlanetaryParameters.csv")
	with p.open("r", newline="") as f:
		reader = csv.reader(f)
		header = next(reader)
		data = []
		for row in reader:
			# convert values to float, empty -> np.nan
			conv = [float(item.strip()) if item.strip() != "" else np.nan for item in row]
			data.append(conv)
	arr = np.array(data, dtype=float)
	return header, arr


# Load planetary parameters into `planetary_params` at import time
header, planetary_params = load_planetary_parameters()

if __name__ == "__main__":
	print("Header:", header)
	print("Planetary parameters array:\n", planetary_params)


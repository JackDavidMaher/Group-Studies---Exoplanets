import numpy as np
import scipy.constants as sc
from scipy.interpolate import RegularGridInterpolator
import astropy.constants as const
import matplotlib.pyplot as plt
import csv

with open('Group 1 Spectrum Code/PlanetaryParameters.csv', mode='r') as file:
    reader = csv.reader(file)
    parameters = {rows[0]: float(rows[1]) for rows in reader}
print(parameters[0])
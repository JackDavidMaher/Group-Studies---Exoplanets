import warnings
warnings.filterwarnings('ignore')
import pandexo.engine.justdoit as jdi # THIS IS THE HOLY GRAIL OF PANDEXO
import numpy as np
import os

exo_dict = jdi.load_exo_dict()
print(exo_dict.keys())


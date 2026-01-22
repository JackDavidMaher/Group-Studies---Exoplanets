from pandeia.engine.calc_utils import build_default_calc
from pandeia.engine.perform_calculation import perform_calculation

# 1. Create a default JWST MIRI imaging setup
calc = build_default_calc('jwst', 'miri', 'imaging')

# 2. Run the engine
report = perform_calculation(calc)

# 3. Check for a result
print(f"Signal-to-Noise Ratio: {report['scalar']['sn']}")

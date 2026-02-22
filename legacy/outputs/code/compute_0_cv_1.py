python
import os
import re

import numpy as np
import pandas as pd

# CV definition:
# CV = | (nA * rA) - 0.5 * (nB * rB + nB' * rB') | / rX

# Synthetic data generation parameters
np.random.seed(42)
num_samples = 100

# Generate synthetic ionic radii (in angstroms, typical ranges)
# rA: A-site cation radius ~1.0 - 1.6 Å
rA = np.random.uniform(1.0, 1.6, num_samples)
# rB: B-site cation radius ~0.5 - 0.9 Å
rB = np.random.uniform(0.5, 0.9, num_samples)
# rB': another B-site cation radius ~0.5 - 0.9 Å
rB_prime = np.random.uniform(0.5, 0.9, num_samples)
# rX: anion radius ~1.2 - 1.4 Å (e.g. O2- ~1.35 Å, halides ~1.2-1.4 Å)
rX = np.random.uniform(1.2, 1.4, num_samples)

# Generate synthetic oxidation states (integer values)
# nA: A-site oxidation state typically +1, +2, +3
nA_choices = np.array([1, 2, 3])
nA = np.random.choice(nA_choices, num_samples)
# nB and nB': B-site oxidation states typically +3, +4, +5
nB_choices = np.array([3, 4, 5])
nB = np.random.choice(nB_choices, num_samples)
nB_prime = np.random.choice(nB_choices, num_samples)
# nX: anion oxidation state typically -2 (O2-) or -1 (halides)
nX_choices = np.array([-2, -1])
nX = np.random.choice(nX_choices, num_samples)

# Compute the CV
CV = np.abs((nA * rA) - 0.5 * (nB * rB + nB_prime * rB_prime)) / rX

# Time variables (t, tau) synthetic generation
# t: simulation time in ps, linearly spaced
t = np.linspace(0, 100, num_samples)
# tau: relaxation time or similar, random between 1 and 10 ps
tau = np.random.uniform(1, 10, num_samples)

# Prepare DataFrame with required columns
df = pd.DataFrame(
    {"sample_id": np.arange(1, num_samples + 1), "rA": rA, "rB": rB, "rX": rX, "t": t, "tau": tau}
)

# Prepare output directory
output_dir = os.path.join("outputs", "csv")
os.makedirs(output_dir, exist_ok=True)

# Construct CSV filename from CV name
cv_name = "Charge-Weighted Octahedral Size Mismatch Index"
# Replace any non-alphanumeric characters with underscore
filename_base = re.sub(r"[^0-9a-zA-Z]+", "_", cv_name).strip("_")
csv_filename = f"{filename_base}.csv"
csv_path = os.path.join(output_dir, csv_filename)

# Save CSV
df.to_csv(csv_path, index=False)

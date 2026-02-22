python
import os
import re

import numpy as np
import pandas as pd

# CV details
cv_name = "Charge-Weighted Octahedral Mismatch Index"

# Prepare output directory
output_dir = os.path.join("outputs", "csv")
os.makedirs(output_dir, exist_ok=True)

# Construct filename from CV name by replacing non-alphanumeric with underscores
filename = re.sub(r"\W+", "_", cv_name) + ".csv"
output_path = os.path.join(output_dir, filename)

# Synthetic data generation parameters
n_samples = 100

# Ionic radii (r) and oxidation states (n) ranges for synthetic data
# Typical ionic radii (in angstroms) ranges:
# A-site cations: ~1.0 - 1.6 Å
# B-site cations: ~0.5 - 0.9 Å
# X-site anions: ~1.2 - 1.5 Å (Oxygen ~1.4, halides ~1.3-1.5)
# Oxidation states (n): integers, typically A: +1 to +3, B: +2 to +6, X: -2 or -1

rng = np.random.default_rng(seed=42)

# Generate synthetic ionic radii
rA = rng.uniform(1.0, 1.6, n_samples)
rB = rng.uniform(0.5, 0.9, n_samples)
rX = rng.uniform(1.2, 1.5, n_samples)

# Generate synthetic oxidation states
# A-site: +1, +2, +3 (choose from these)
nA = rng.choice([1, 2, 3], n_samples)
# B-site: +2, +3, +4, +5, +6
nB = rng.choice([2, 3, 4, 5, 6], n_samples)
# X-site: -2 (oxide) or -1 (halide)
nX = rng.choice([-2, -1], n_samples)

# Compute CV for ABX3 (single perovskite)
# CV = (n_A * r_A + n_B * r_B) / (|n_A * r_A - n_B * r_B| + r_X)

numerator = nA * rA + nB * rB
denominator = np.abs(nA * rA - nB * rB) + rX
cv_values = numerator / denominator

# Prepare output DataFrame with required columns:
# sample_id, rA, rB, rX, t, tau
# The problem does not define t and tau; we generate synthetic values for demonstration:
# Let's define t as the tolerance factor (Goldschmidt tolerance factor) for perovskites:
# t = (rA + rX) / (np.sqrt(2) * (rB + rX))
t = (rA + rX) / (np.sqrt(2) * (rB + rX))

# tau: a synthetic parameter related to charge mismatch, e.g. |nA - nB| / (|nX| + 1)
tau = np.abs(nA - nB) / (np.abs(nX) + 1)

df = pd.DataFrame(
    {"sample_id": np.arange(1, n_samples + 1), "rA": rA, "rB": rB, "rX": rX, "t": t, "tau": tau}
)

# Save to CSV
df.to_csv(output_path, index=False)

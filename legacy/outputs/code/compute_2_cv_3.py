python
import os
import re

import numpy as np
import pandas as pd

# CV metadata
cv_name = "Modified Geometric Stability Factor (MGSF)"


# Function to sanitize filename from CV name
def sanitize_filename(name):
    # Replace any character not alphanumeric or underscore with underscore
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") + ".csv"


# Function to compute MGSF for single perovskite
def compute_mgsf_single(rA, rB, rX, nA, nB):
    numerator = rA + rX
    denominator = np.sqrt(rB + rX)
    exponent = -np.abs(nA / rA - nB / rB)
    return (numerator / denominator) * np.exp(exponent)


# Function to compute MGSF for double perovskite
def compute_mgsf_double(rA, rB, rBp, rX, nA, nB, nBp):
    numerator = rA + rX
    denominator = np.sqrt((rB + rBp) / 2 + rX)
    exponent = -np.abs(nA / rA - (nB + nBp) / (2 * rB))
    return (numerator / denominator) * np.exp(exponent)


# Generate synthetic data
# Let's create 50 samples: half single perovskites, half double perovskites

num_samples = 50
half = num_samples // 2

# Ionic radii (in angstroms) typical ranges for perovskites:
# A-site: 1.0 - 1.6 (e.g. Cs+, Rb+, K+)
# B-site: 0.5 - 0.9 (e.g. Ti4+, Fe3+, Sn4+)
# B'-site: same range as B-site for double perovskites
# X-site: 1.2 - 1.9 (O2-, Cl-, Br-, I-)

np.random.seed(42)  # reproducible

# Single perovskite data
rA_single = np.random.uniform(1.0, 1.6, half)
rB_single = np.random.uniform(0.5, 0.9, half)
rX_single = np.random.uniform(1.2, 1.9, half)

# Oxidation states typical for perovskites:
# A-site: +1 or +2 (e.g. Cs+, Sr2+)
# B-site: +3 or +4 (e.g. Fe3+, Ti4+)
# X-site: -2 for oxides, -1 for halides (we'll pick -2 or -1 randomly)

nA_single = np.random.choice([1, 2], half)
nB_single = np.random.choice([3, 4], half)
nX_single = np.random.choice([-2, -1], half)

# Double perovskite data
rA_double = np.random.uniform(1.0, 1.6, half)
rB_double = np.random.uniform(0.5, 0.9, half)
rBp_double = np.random.uniform(0.5, 0.9, half)
rX_double = np.random.uniform(1.2, 1.9, half)

nA_double = np.random.choice([1, 2], half)
nB_double = np.random.choice([3, 4], half)
nBp_double = np.random.choice([3, 4], half)
nX_double = np.random.choice([-2, -1], half)

# Compute MGSF values
mgsf_single = compute_mgsf_single(rA_single, rB_single, rX_single, nA_single, nB_single)
mgsf_double = compute_mgsf_double(
    rA_double, rB_double, rBp_double, rX_double, nA_double, nB_double, nBp_double
)

# Compute tolerance factor t = (rA + rX) / (sqrt(2)*(rB + rX)) for single perovskite (common definition)
t_single = (rA_single + rX_single) / (np.sqrt(2) * (rB_single + rX_single))
# For double perovskite, average B radii
t_double = (rA_double + rX_double) / (np.sqrt(2) * ((rB_double + rBp_double) / 2 + rX_double))

# Compute tau = (rA + rX) / (rB + rX) for single perovskite (a simple size ratio)
tau_single = (rA_single + rX_single) / (rB_single + rX_single)
tau_double = (rA_double + rX_double) / ((rB_double + rBp_double) / 2 + rX_double)

# Prepare dataframes
df_single = pd.DataFrame(
    {
        "sample_id": [f"S_{i + 1}" for i in range(half)],
        "rA": rA_single,
        "rB": rB_single,
        "rX": rX_single,
        "t": t_single,
        "tau": tau_single,
    }
)

df_double = pd.DataFrame(
    {
        "sample_id": [f"D_{i + 1}" for i in range(half)],
        "rA": rA_double,
        "rB": (rB_double + rBp_double) / 2,
        "rX": rX_double,
        "t": t_double,
        "tau": tau_double,
    }
)

# Combine data
df = pd.concat([df_single, df_double], ignore_index=True)

# Create output directory
output_dir = os.path.join("outputs", "csv")
os.makedirs(output_dir, exist_ok=True)

# Construct filename
filename = sanitize_filename(cv_name)
filepath = os.path.join(output_dir, filename)

# Save to CSV with specified columns
df.to_csv(filepath, columns=["sample_id", "rA", "rB", "rX", "t", "tau"], index=False)

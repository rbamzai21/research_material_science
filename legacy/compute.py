import numpy as np
import pandas as pd

INPUT_PATH = "perovskites.csv"
OUTPUT_PATH = "pervoskites_with_cvs.csv"

df = pd.read(INPUT_PATH)


def avg_B_radius(row):
    if not pd.isna(row["r_Bp"]):
        return 0.5 * (row["r_B"] + row["r_Bp"])
    return row["r_B"]


def avg_B_charge(row):
    if not pd.isna(row["z_Bp"]):
        return 0.5 * (abs(row["z_B"]) + abs(row["z_Bp"]))
    return abs(row["z_B"])


df["t_goldschmidt"] = (df["r_A"] + df["r_X"]) / (np.sqrt(2) * (df["r_B"] + df["r_X"]))

df["octahedral_factor"] = df["r_B"] / df["r_X"]

df["radius_ratio"] = df["r_A"] / df["r_B"]

df["t_double"] = (df["r_A"] + df["r_X"]) / (
    np.sqrt(2) * (df.apply(avg_B_radius, axis=1) + df["r_X"])
)

df["t_charge_corrected"] = df["t_goldschmidt"] * (df.apply(avg_B_charge, axis=1) / abs(df["z_A"]))

df.to_csv(OUTPUT_PATH, index=False)

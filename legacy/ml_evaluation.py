import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

CSV_DIR = "outputs/csv"

# Loop over all CSV files
for csv_file in sorted(os.listdir(CSV_DIR)):
    if not csv_file.endswith(".csv"):
        continue

    file_path = os.path.join(CSV_DIR, csv_file)
    df = pd.read_csv(file_path)

    cv_name = os.path.splitext(csv_file)[0]
    print(f"\nProcessing CV: {cv_name}")

    # Simple linear regression: tau vs t
    if "t" in df.columns and "tau" in df.columns:
        X = df[["t"]].values
        y = df["tau"].values

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        r2 = r2_score(y, y_pred)
        a = model.coef_[0]
        b = model.intercept_

        print(f"tau = {a:.4f} * t + {b:.4f}")
        print(f"R^2 = {r2:.4f}")

        # Plot
        plt.figure()
        plt.scatter(X, y, label="Data")
        plt.plot(X, y_pred, label="Fit", color="red")
        plt.xlabel("t")
        plt.ylabel("tau")
        plt.title(cv_name)
        plt.legend()
        plt.show()

    # Multi-variable regression: tau vs rA, rB, rX, t
    if all(col in df.columns for col in ["rA", "rB", "rX", "t", "tau"]):
        X = df[["rA", "rB", "rX", "t"]].values
        y = df["tau"].values

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)

        print("Multi-variable regression:")
        print("R^2:", r2_score(y, y_pred))
        print("Coefficients:", model.coef_)
        print("Intercept:", model.intercept_)

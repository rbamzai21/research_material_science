from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

X = df[[
    "t_goldschmidt", 
    "octahedral_factor",
    "radius_ratio",
    "t_double",
    "t_charge_corrected"
]]

y = df["structure_label"]
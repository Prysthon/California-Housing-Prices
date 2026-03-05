import joblib
import pandas as pd

MODEL_PATH = "modelos/ridge_poly_target_quantile.joblib"

model = joblib.load(MODEL_PATH)
print("Loaded:", type(model))
print("Has predict:", hasattr(model, "predict"))
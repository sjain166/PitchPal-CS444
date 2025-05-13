import joblib
label_encoder = joblib.load("label_encoder.pkl")
print("All labels:", label_encoder.classes_)
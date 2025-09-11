import streamlit as st
import joblib
import numpy as np

model = joblib.load('model.pkl')

st.title("Cancer Prediction App")

default_features = {
    'radius_mean': 14.127,
    'texture_mean': 19.289,
    'perimeter_mean': 91.969,
    'area_mean': 654.889,
    'smoothness_mean': 0.09636,
    'compactness_mean': 0.10434,
    'concavity_mean': 0.0888,
    'concave points_mean': 0.04891,
    'symmetry_mean': 0.1812,
    'fractal_dimension_mean': 0.0628,
    'radius_se': 0.40517,
    'texture_se': 1.21685,
    'perimeter_se': 2.86606,
    'area_se': 40.337,
    'smoothness_se': 0.006414,
    'compactness_se': 0.025478,
    'concavity_se': 0.031894,
    'concave points_se': 0.011796,
    'symmetry_se': 0.020542,
    'fractal_dimension_se': 0.003794,
    'radius_worst': 16.26919,
    'texture_worst': 25.677223,
    'perimeter_worst': 107.261292,
    'area_worst': 880.583128,
    'smoothness_worst': 0.132368,
    'compactness_worst': 0.254265,
    'concavity_worst': 0.272188,
    'concave points_worst': 0.114606,
    'symmetry_worst': 0.290076,
    'fractal_dimension_worst': 0.083946
}

radius_se = st.number_input("Radius SE", value=float(default_features["radius_se"]))
symmetry_worst = st.number_input("Symmetry Worst", value=float(default_features["symmetry_worst"]))
texture_worst = st.number_input("Texture Worst", value=float(default_features["texture_worst"]))
concave_points_se = st.number_input("Concave Ponits SE", value=float(default_features["concave points_se"]))
concavity_worst = st.number_input("Concavity Worst", value=float(default_features["concavity_worst"]))

feature_order = [
    'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
    'compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
    'compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se',
    'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
    'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'
]

input_features = []
for f in feature_order:
    if f == 'radius_se':
        input_features.append(radius_se)
    elif f == 'symmetry_worst':
        input_features.append(symmetry_worst)
    elif f == 'texture_worst':
        input_features.append(texture_worst)
    elif f == 'concave points_se':
        input_features.append(concave_points_se)
    elif f == 'concavity_worst':
        input_features.append(concavity_worst)
    else:
        input_features.append(float(default_features[f]))

input_features = np.array([input_features])

if st.button("Predict"):
    prediction = model.predict(input_features)
    final_prediction = "Benign" if prediction[0]==1 else "Malignant"
    st.success(f"Predicted Class: {final_prediction}")

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the best model (update the filename if needed)
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/best_model_GradientBoosting.joblib')
model = joblib.load(MODEL_PATH)

# For feature importances (if available)
def get_feature_importances(model, feature_names):
	if hasattr(model, 'feature_importances_'):
		return dict(zip(feature_names, model.feature_importances_))
	elif hasattr(model, 'coef_'):
		return dict(zip(feature_names, np.abs(model.coef_).flatten()))
	else:
		return {}

# Risk tier logic
def risk_tier(prob):
	if prob >= 0.7:
		return 'High'
	elif prob >= 0.4:
		return 'Medium'
	else:
		return 'Low'


API_KEY = "mysecretapikey"  # Change this in production!

@app.route('/predict', methods=['POST'])
def predict():
	# Check API key in headers
	req_key = request.headers.get('x-api-key')
	if req_key != API_KEY:
		return jsonify({"error": "Unauthorized. Invalid API key."}), 401

	data = request.json
	# Convert input to DataFrame
	input_df = pd.DataFrame([data])
	# Feature engineering and cleaning
	from importlib.util import spec_from_file_location, module_from_spec
	fe_path = os.path.join(os.path.dirname(__file__), '../data/feature_engineering.py')
	spec = spec_from_file_location('feature_engineering', fe_path)
	fe = module_from_spec(spec)
	spec.loader.exec_module(fe)
	input_df = fe.clean_data(input_df)
	input_df = fe.engineer_features(input_df)
	# Prepare features as in training
	drop_cols = ['CustomerID', 'Churn', 'Last_Interaction']
	for col in drop_cols:
		if col in input_df.columns:
			input_df = input_df.drop(columns=[col])
	# One-hot encode
	input_df = pd.get_dummies(input_df, drop_first=True)
	# Drop datetime columns
	datetime_cols = input_df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
	input_df = input_df.drop(columns=datetime_cols)
	# Fill missing columns with 0 (align with model)
	model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else input_df.columns
	for col in model_features:
		if col not in input_df.columns:
			input_df[col] = 0
	input_df = input_df[model_features]
	# Predict
	prob = float(model.predict_proba(input_df)[0][1])
	tier = risk_tier(prob)
	# Feature importances (top 3)
	importances = get_feature_importances(model, model_features)
	top_features = sorted(importances.items(), key=lambda x: -x[1])[:3]
	return jsonify({
		'churn_probability': prob,
		'risk_tier': tier,
		'top_features': top_features
	})

@app.route('/')
def health():
	return 'Churn Prediction API is running.'

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, debug=True)

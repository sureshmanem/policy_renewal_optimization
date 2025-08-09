from flask import Flask, request, jsonify

# --- Flask App Instance ---
app = Flask(__name__)
import joblib
import pandas as pd
import numpy as np

import logging
import csv
from datetime import datetime
import os


# --- Logging Setup ---
logging.basicConfig(
	filename=os.path.join(os.path.dirname(__file__), 'api_requests.log'),
	level=logging.INFO,
	format='%(asctime)s %(levelname)s %(message)s'
)

FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), 'feedback_log.csv')

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
	req_key = request.headers.get('x-api-key')
	if req_key != API_KEY:
		logging.warning(f"Unauthorized API key: {req_key}")
		return jsonify({"error": "Unauthorized. Invalid API key."}), 401
	data = request.json
	try:
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
		logging.info(f"Prediction request: input={data} output={{'prob': {prob}, 'tier': '{tier}'}}")
		return jsonify({
			'churn_probability': prob,
			'risk_tier': tier,
			'top_features': top_features
		})
	except Exception as e:
		logging.error(f"Error during prediction: {e}")
		return jsonify({"error": str(e)}), 500

# --- Feedback endpoint ---
@app.route('/feedback', methods=['POST'])
def feedback():
	data = request.json
	# Add timestamp to feedback
	data['timestamp'] = datetime.now().isoformat()
	# Append feedback to CSV
	fieldnames = list(data.keys())
	file_exists = os.path.isfile(FEEDBACK_FILE)
	with open(FEEDBACK_FILE, 'a', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		if not file_exists:
			writer.writeheader()
		writer.writerow(data)
	logging.info(f"Feedback received: {data}")
	return jsonify({"status": "success", "message": "Feedback logged. Thank you!"})

@app.route('/')
def health():
	return 'Churn Prediction API is running.'

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=5000, debug=True)

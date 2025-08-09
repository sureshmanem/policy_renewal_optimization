# Model training script for policy renewal churn prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import joblib
import os

# Optional: CatBoost and LightGBM
try:
	from catboost import CatBoostClassifier
	catboost_available = True
except ImportError:
	catboost_available = False
try:
	from lightgbm import LGBMClassifier
	lightgbm_available = True
except ImportError:
	lightgbm_available = False

def load_data(filepath):
	return pd.read_csv(filepath)

def prepare_data(df):
	# Target: Churn (1 if churned, 0 if retained)
	df = df.copy()
	# Ensure 'Churn' is numeric (if not, map values accordingly)
	if df['Churn'].dtype == object:
		df['target'] = df['Churn'].map({'Yes': 1, 'No': 0, '1': 1, '0': 0})
	else:
		df['target'] = df['Churn']
	df = df.dropna(subset=['target'])
	# Drop columns not needed for features
	drop_cols = ['CustomerID', 'Churn', 'Last Interaction']
	X = df.drop(columns=[c for c in drop_cols if c in df.columns] + ['target'])
	y = df['target']
	# One-hot encode categorical columns
	X = pd.get_dummies(X, drop_first=True)
	# Drop datetime columns
	datetime_cols = X.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
	X = X.drop(columns=datetime_cols)
	# Fill NaNs
	X = X.fillna(0)
	# Remove inf/-inf
	X = X.replace([np.inf, -np.inf], 0)
	# Scale numeric features
	from sklearn.preprocessing import StandardScaler
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)
	X = pd.DataFrame(X_scaled, columns=X.columns)
	return X, y

def train_and_evaluate(X_train, X_test, y_train, y_test):
	models = {
		'LogisticRegression': LogisticRegression(max_iter=1000),
		'RandomForest': RandomForestClassifier(),
		'GradientBoosting': GradientBoostingClassifier()
	}
	if catboost_available:
		models['CatBoost'] = CatBoostClassifier(verbose=0)
	if lightgbm_available:
		models['LightGBM'] = LGBMClassifier()

	results = {}
	for name, model in models.items():
		model.fit(X_train, y_train)
		y_pred = model.predict_proba(X_test)[:,1]
		auc = roc_auc_score(y_test, y_pred)
		results[name] = {'model': model, 'auc': auc}
		print(f"{name} AUC-ROC: {auc:.3f}")
	return results

if __name__ == "__main__":
	# Load engineered data
	data_path = os.path.join('..', 'data', 'training_data.csv')
	from importlib.util import spec_from_file_location, module_from_spec
	fe_path = os.path.join('..', 'data', 'feature_engineering.py')
	spec = spec_from_file_location('feature_engineering', fe_path)
	fe = module_from_spec(spec)
	spec.loader.exec_module(fe)
	df = fe.load_data(data_path)
	df = fe.clean_data(df)
	df = fe.engineer_features(df)

	X, y = prepare_data(df)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
	results = train_and_evaluate(X_train, X_test, y_train, y_test)

	# Save best model
	best_model_name = max(results, key=lambda k: results[k]['auc'])
	best_model = results[best_model_name]['model']
	joblib.dump(best_model, f"best_model_{best_model_name}.joblib")
	print(f"Best model: {best_model_name} (AUC-ROC: {results[best_model_name]['auc']:.3f}) saved as best_model_{best_model_name}.joblib")

# Data Preparation & Feature Engineering for Policy Renewal Optimization
import pandas as pd
import numpy as np
from datetime import datetime

def load_data(filepath):
	"""Load data from CSV file."""
	return pd.read_csv(filepath)

def clean_data(df):
	"""Basic cleaning: handle missing values, correct types."""
	# Example: fill missing numeric with median, categorical with mode
	for col in df.select_dtypes(include=[np.number]).columns:
		df[col] = df[col].fillna(df[col].median())
	for col in df.select_dtypes(include=[object]).columns:
		df[col] = df[col].fillna(df[col].mode()[0])
	# Convert dates
	if 'renewal_date' in df.columns:
		df['renewal_date'] = pd.to_datetime(df['renewal_date'], errors='coerce')
	return df

def engineer_features(df):
	"""Create new features as per requirements."""
	# Premium change % since last renewal (requires previous premium column)
	if 'premium_amount' in df.columns and 'previous_premium_amount' in df.columns:
		df['premium_change_pct'] = (df['premium_amount'] - df['previous_premium_amount']) / df['previous_premium_amount']
	else:
		df['premium_change_pct'] = np.nan

	# Time since last claim (requires last_claim_date column)
	if 'last_claim_date' in df.columns:
		df['last_claim_date'] = pd.to_datetime(df['last_claim_date'], errors='coerce')
		df['days_since_last_claim'] = (datetime.now() - df['last_claim_date']).dt.days
	else:
		df['days_since_last_claim'] = np.nan

	# Average claim amount
	if 'total_claim_amount' in df.columns and 'num_claims' in df.columns:
		df['avg_claim_amount'] = df['total_claim_amount'] / df['num_claims'].replace(0, np.nan)
	else:
		df['avg_claim_amount'] = np.nan

	# Customer loyalty score (example: tenure * on-time payment ratio)
	if 'customer_tenure' in df.columns and 'on_time_payments' in df.columns and 'delayed_payments' in df.columns and 'missed_payments' in df.columns:
		total_payments = df['on_time_payments'] + df['delayed_payments'] + df['missed_payments']
		payment_ratio = df['on_time_payments'] / total_payments.replace(0, np.nan)
		df['loyalty_score'] = df['customer_tenure'] * payment_ratio
	else:
		df['loyalty_score'] = np.nan

	# Risk tier placeholder (to be set by model)
	df['risk_tier'] = 'Unknown'

	return df

if __name__ == "__main__":
	# Example usage
	df = load_data("sample_schema.csv")
	df = clean_data(df)
	df = engineer_features(df)
	print(df.head())

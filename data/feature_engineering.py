# Feature engineering for new training_data.csv schema
import pandas as pd
import numpy as np

def load_data(filepath):
	"""Load data from CSV file."""
	return pd.read_csv(filepath)

def clean_data(df):
	"""Basic cleaning: handle missing values, correct types."""
	# Convert all columns that should be numeric
	numeric_cols = ['Age', 'Tenure', 'Usage_Frequency', 'Support_Calls', 'Payment_Delay', 'Contract_Length', 'Total_Spend']
	for col in numeric_cols:
		if col in df.columns:
			df[col] = pd.to_numeric(df[col], errors='coerce')
	# Fill missing numeric with median, categorical with mode
	for col in df.select_dtypes(include=[np.number]).columns:
		df[col] = df[col].fillna(df[col].median())
	for col in df.select_dtypes(include=[object]).columns:
		df[col] = df[col].fillna(df[col].mode()[0])
	return df

def engineer_features(df):
	"""Feature engineering for new schema."""
	# Example engineered features for the new schema
	# Usage per tenure
	if 'Usage_Frequency' in df.columns and 'Tenure' in df.columns:
		df['Usage_per_Tenure'] = df['Usage_Frequency'] / (df['Tenure'] + 1)
	# Payment delay ratio
	if 'Payment_Delay' in df.columns and 'Contract_Length' in df.columns:
		df['Payment_Delay_Ratio'] = df['Payment_Delay'] / (df['Contract_Length'] + 1)
	# Support calls per tenure
	if 'Support_Calls' in df.columns and 'Tenure' in df.columns:
		df['Support_Calls_per_Tenure'] = df['Support_Calls'] / (df['Tenure'] + 1)
	# Spend per month
	if 'Total_Spend' in df.columns and 'Contract_Length' in df.columns:
		df['Spend_per_Month'] = df['Total_Spend'] / (df['Contract_Length'] + 1)
	# Days since last interaction (if Last_Interaction is a date)
	if 'Last_Interaction' in df.columns:
		try:
			df['Last_Interaction'] = pd.to_datetime(df['Last_Interaction'], errors='coerce')
			df['Days_Since_Last_Interaction'] = (pd.Timestamp.now() - df['Last_Interaction']).dt.days
		except Exception:
			df['Days_Since_Last_Interaction'] = np.nan
	return df

if __name__ == "__main__":
	df = load_data("training_data.csv")
	df = clean_data(df)
	df = engineer_features(df)
	print(df.head())

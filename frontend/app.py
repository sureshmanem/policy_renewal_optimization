



import streamlit as st
import requests
import pandas as pd



st.set_page_config(page_title="Policy Churn Prediction", layout="wide")

st.markdown("""
<h2 style='text-align:center;'>Policy Renewal Churn Prediction</h2>
<p style='text-align:center;'>Enter customer and policy details to get a comprehensive churn risk prediction, actionable insights, and explanations for the result.</p>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,1,1])
with col1:
	CustomerID = st.text_input("Customer ID", key="cid")
	Age = st.number_input("Age", min_value=0, max_value=120, value=30, key="age")
	Gender = st.selectbox("Gender", ["Male", "Female", "Other"], key="gender")
	Tenure = st.number_input("Tenure (months)", min_value=0, value=12, key="tenure")
with col2:
	Usage_Frequency = st.number_input("Usage Frequency", min_value=0, value=5, key="usage")
	Support_Calls = st.number_input("Support Calls", min_value=0, value=0, key="support")
	Payment_Delay = st.number_input("Payment Delay (days)", min_value=0, value=0, key="delay")
	Subscription_Type = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"], key="subtype")
with col3:
	Contract_Length = st.number_input("Contract Length (months)", min_value=1, value=12, key="contract")
	Total_Spend = st.number_input("Total Spend", min_value=0.0, value=1000.0, key="spend")
	Last_Interaction = st.date_input("Last Interaction", key="lastint")
	submitted = st.button("Predict Churn", use_container_width=True)

def get_risk_message(tier):
	if tier == 'High':
		return ("<span style='color:red'><b>High Risk:</b> Very likely to churn. Immediate retention action is recommended.")
	elif tier == 'Medium':
		return ("<span style='color:orange'><b>Medium Risk:</b> Moderate churn risk. Monitor and consider proactive engagement.")
	else:
		return ("<span style='color:green'><b>Low Risk:</b> Likely to renew. Maintain engagement.")

def get_probability_message(prob):
	if prob >= 0.7:
		return "This customer is at significant risk of not renewing."
	elif prob >= 0.4:
		return "There is a meaningful risk of churn, but retention is possible with the right actions."
	else:
		return "The customer is likely to stay, but continue to provide good service."

if submitted:
	input_data = {
		"CustomerID": CustomerID,
		"Age": Age,
		"Gender": Gender,
		"Tenure": Tenure,
		"Usage_Frequency": Usage_Frequency,
		"Support_Calls": Support_Calls,
		"Payment_Delay": Payment_Delay,
		"Subscription_Type": Subscription_Type,
		"Contract_Length": Contract_Length,
		"Total_Spend": Total_Spend,
		"Last_Interaction": str(Last_Interaction),
	}
	try:
		headers = {"x-api-key": "mysecretapikey"}  # Must match backend
		response = requests.post("http://localhost:5000/predict", json=input_data, headers=headers)
		if response.status_code == 200:
			result = response.json()
			prob = result['churn_probability']
			tier = result['risk_tier']
			st.markdown(f"<h3 style='text-align:center;'>Churn Probability: <span style='color:blue'>{prob:.2f}</span></h3>", unsafe_allow_html=True)
			st.markdown(f"<div style='text-align:center;'>{get_risk_message(tier)}</div>", unsafe_allow_html=True)
			st.info(get_probability_message(prob))
			st.markdown("<hr>", unsafe_allow_html=True)
			st.markdown("**Key Drivers of Prediction:**")
			st.caption("These are the most influential features for this specific prediction. Higher absolute values indicate a stronger impact on the churn risk. Positive values increase churn risk, negative values decrease it. Use these insights to guide your retention strategy.")
			for feat, val in result['top_features']:
				st.write(f"- **{feat}**: {val:.3f}")
			st.markdown("<hr>", unsafe_allow_html=True)
			st.caption("Model predictions are for guidance. Use your expertise and context for final decisions.")

			# --- Feedback Button ---
			if st.button("Flag this prediction as uncertain or incorrect"):
				feedback_data = input_data.copy()
				feedback_data['churn_probability'] = prob
				feedback_data['risk_tier'] = tier
				feedback_data['top_features'] = str(result['top_features'])
				try:
					headers = {"x-api-key": "mysecretapikey"}
					resp = requests.post("http://localhost:5000/feedback", json=feedback_data, headers=headers)
					if resp.status_code == 200:
						st.success("Thank you for your feedback! This case will be reviewed.")
					else:
						st.error(f"Feedback error: {resp.text}")
				except Exception as e:
					st.error(f"Could not send feedback: {e}")
		else:
			st.error(f"API Error: {response.text}")
	except Exception as e:
		st.error(f"Could not connect to prediction API: {e}")

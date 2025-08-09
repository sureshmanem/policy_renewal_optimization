
# Policy Renewal Optimization – Churn Prediction System

## Overview
This project is a full-stack machine learning system for predicting the likelihood that an insurance policyholder will NOT renew their policy (churn). It enables customer service agents to input customer and policy details via a web UI and receive real-time churn predictions, risk tiers, and key drivers. The system also collects user feedback for continuous improvement.

## Features
- **Machine Learning Model**: Trained with scikit-learn, supports feature engineering and one-hot encoding.
- **API Backend**: Flask-based, serves predictions and collects feedback, with API key security and logging.
- **Frontend UI**: Streamlit app for agent-friendly data entry, prediction display, and feedback submission.
- **Monitoring & Feedback**: Logs all API requests and user feedback for monitoring and model improvement.

## Project Structure
```
policy_renewal_optimization/
├── api/
│   └── app.py                # Flask API backend
├── frontend/
│   └── app.py                # Streamlit frontend UI
├── data/
│   └── feature_engineering.py# Feature engineering functions
├── model/
│   └── train_model.py        # Model training script
│   └── best_model_GradientBoosting.joblib # Trained model
├── feedback_log.csv          # User feedback log
├── api_requests.log          # API request log
└── README.md                 # Project documentation
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repo-url>
cd policy_renewal_optimization
```

### 2. Create and Activate Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
If `requirements.txt` is missing, install manually:
```bash
pip install flask streamlit pandas numpy scikit-learn joblib
```

### 4. Train the Model (Optional)
If you want to retrain the model:
```bash
python model/train_model.py
```

### 5. Start the Backend API
```bash
python api/app.py
```
The API will run at `http://localhost:5000`.

### 6. Start the Frontend UI
```bash
streamlit run frontend/app.py
```
The UI will be available at the URL shown in the terminal (usually `http://localhost:8501`).

## Usage
1. Open the Streamlit UI in your browser.
2. Enter customer and policy details in the form.
3. Click **Predict** to get churn probability, risk tier, and key drivers.
4. If the prediction seems incorrect or uncertain, use the **feedback** button to flag it for review.

## API Endpoints
- `POST /predict` – Get churn prediction. Requires `x-api-key` header.
- `POST /feedback` – Submit feedback on a prediction. Requires `x-api-key` header.
- `GET /` – Health check.

## Security
- The API requires an API key (`x-api-key: mysecretapikey`). Change this key in production.

## Monitoring & Feedback
- All API requests are logged in `api_requests.log`.
- User feedback is logged in `feedback_log.csv`.

## Customization & Productionization
- Update the model or feature engineering as needed.
- For production, use environment variables for secrets and consider Dockerizing the app.
- Add unit tests for robustness.

## Troubleshooting
- If the UI cannot connect to the API, ensure the backend is running and accessible at `localhost:5000`.
- Check `api_requests.log` for errors.
- Ensure all dependencies are installed in your virtual environment.

## License
MIT License

---
For questions or contributions, please open an issue or pull request.

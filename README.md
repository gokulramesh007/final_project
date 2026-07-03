# Loan Approval Prediction Frontend (Flask)

## What it does
Web UI that collects the loan inputs (excluding `Loan_ID`) and uses `loan_approval_prediction.pkl` (RandomForest) to predict:

- **Approved** (1)
- **Rejected** (0)

## Files
- `app.py` - Flask backend + model loading
- `templates/index.html` - Frontend form + result display
- `requirements.txt` - Python dependencies

## Run locally
1. Open a terminal in this folder:
   ```bash
   cd d:/ai_ml/loan-approval-frontend
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Copy the model file into this folder (if it isn’t already):
   - `d:/ai_ml/loan_approval_prediction.pkl` → `d:/ai_ml/loan-approval-frontend/loan_approval_prediction.pkl`

4. Start the server:
   ```bash
   python app.py
   ```

5. Open:
   - http://127.0.0.1:5000

## Note about feature names
This project assumes the model expects one-hot encoded columns.
If you get a prediction error about missing columns, the model was trained with different preprocessing.
In that case, we should load the training pipeline or replicate the exact preprocessing steps.


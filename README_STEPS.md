# What to expect
This Flask UI uses the saved sklearn model from `ML Project 2 - Loan Approval Prediction.pkl`.

## Why it keeps rejecting
The loaded model expects features like:
- `cibil_score`
- `residential_assets_value`
- `commercial_assets_value`
- `luxury_assets_value`
- `bank_asset_value`

But the current UI only collects:
- ApplicantIncome, CoapplicantIncome
- LoanAmount, Loan_Amount_Term
- Credit_History, Total_Income
- and a few categorical fields

In `app.py`, those missing fields are currently defaulted to `0`, which can easily make the model predict **Rejected** every time.

## Next fix (required)
Add the missing fields to `templates/index.html` and map them in `app.py` so the DataFrame columns match exactly the model's `feature_names_in_`.


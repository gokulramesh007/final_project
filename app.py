from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd


app = Flask(__name__)
MODEL_PATH = "loan_approval_prediction.pkl"

# If the expected model filename isn't found inside this folder,
# fall back to the repo-level model file.
import os

if not os.path.exists(MODEL_PATH):
    # repo model filename is: ML Project 2 - Loan Approval Prediction.pkl
    fallback = os.path.join(os.path.dirname(__file__), "..", "ML Project 2 - Loan Approval Prediction.pkl")
    fallback = os.path.normpath(fallback)
    if os.path.exists(fallback):
        MODEL_PATH = fallback
    else:
        # last resort: load the pkl with the exact filename in repo root
        MODEL_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "ML Project 2 - Loan Approval Prediction.pkl"))



model = joblib.load(MODEL_PATH)

# Expected input columns for the model after your preprocessing
EXPECTED_COLUMNS = [
    'gender_Male',
    'self_employed',
    'married_Yes',
    'dependents_0',
    'dependents_1',
    'dependents_2',
    'dependents_3+',
    'education_Graduate',
    'education_Not Graduate',
    'applicant_income',
    'coapplicant_income',
    'loan_amount',
    'loan_term',
    'credit_history',
    'property_area_Urban',
    'property_area_Semiurban',
    'property_area_Rural',
    'total_income',
    'loan_amount_log',
    'total_income_log',
]


def _to_float(x):

    try:
        if x is None:
            return 0.0
        return float(x)
    except Exception:
        return 0.0



def _prepare_features(form):
    """Build a single-row DataFrame whose columns match what the loaded model expects."""

    # ---- Categorical one-hot / dummy encoding ----
    gender = form.get('gender')  # Male/Female
    gender_Male = 1 if gender == 'Male' else 0

    self_employed = 1 if form.get('Self_Employed') == 'Yes' else 0

    married = form.get('married')  # Yes/No
    married_Yes = 1 if married == 'Yes' else 0

    dependents_raw = form.get('dependents')  # expected: 0/1/2/3+
    if dependents_raw == '3+':
        dependents_val = None
    else:
        dependents_val = int(dependents_raw) if dependents_raw is not None else 0

    dependents_0 = 1 if dependents_val == 0 else 0
    dependents_1 = 1 if dependents_val == 1 else 0
    dependents_2 = 1 if dependents_val == 2 else 0
    dependents_3_plus = 1 if dependents_raw == '3+' else 0


    education = form.get('education')  # Graduate/Not Graduate
    education_Graduate = 1 if education == 'Graduate' else 0
    education_Not_Graduate = 1 if education == 'Not Graduate' else 0

    property_area = form.get('property_area')  # Urban/Semiurban/Rural
    property_area_Urban = 1 if property_area == 'Urban' else 0
    property_area_Semiurban = 1 if property_area == 'Semiurban' else 0
    property_area_Rural = 1 if property_area == 'Rural' else 0

    # ---- Numeric fields ----
    applicant_income = _to_float(form.get('income_annum'))
    coapplicant_income = _to_float(form.get('coapplicant_income'))
    total_income = applicant_income + coapplicant_income

    loan_amount = _to_float(form.get('loan_amount'))
    loan_term = _to_float(form.get('loan_term'))
    credit_history = _to_float(form.get('credit_history'))
    cibil_score = _to_float(form.get('cibil_score'))

    residential_assets_value = _to_float(form.get('residential_assets_value'))
    commercial_assets_value = _to_float(form.get('commercial_assets_value'))
    luxury_assets_value = _to_float(form.get('luxury_assets_value'))
    bank_asset_value = _to_float(form.get('bank_asset_value'))

    # Logs (auto-calc if not provided)
    loan_amount_log = _to_float(form.get('loan_amount_log'))
    if form.get('loan_amount_log') in (None, '',):
        loan_amount_log = float(np.log1p(loan_amount))

    total_income_log = _to_float(form.get('total_income_log'))
    if form.get('total_income_log') in (None, '',):
        total_income_log = float(np.log1p(total_income))

    # Build baseline row with canonical feature names.
    row = {
        'gender_Male': gender_Male,
        'self_employed': self_employed,
        'married_Yes': married_Yes,
        'dependents_0': dependents_0,
        'dependents_1': dependents_1,
        'dependents_2': dependents_2,
        'dependents_3+': dependents_3_plus,
        'education_Graduate': education_Graduate,
        'education_Not Graduate': education_Not_Graduate,
        'applicant_income': applicant_income,
        'coapplicant_income': coapplicant_income,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'credit_history': credit_history,
        'property_area_Urban': property_area_Urban,
        'property_area_Semiurban': property_area_Semiurban,
        'property_area_Rural': property_area_Rural,
        'total_income': total_income,
        'loan_amount_log': loan_amount_log,
        'total_income_log': total_income_log,
        # if model expects this original name, keep it consistent:
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value,
        # Some notebook variants may include an alternate education dummy name:
        'education_Not Graduate ': education_Not_Graduate,
    }

    x = pd.DataFrame([row])

    # Align to model expectations if available
    model_feature_names = getattr(model, "feature_names_in_", None)
    if model_feature_names is not None:
        for c in model_feature_names:
            if c not in x.columns:
                x[c] = 0
        x = x[list(model_feature_names)]
        return x

    # Fallback: align to EXPECTED_COLUMNS if model doesn't expose feature_names_in_
    for c in EXPECTED_COLUMNS:
        if c not in x.columns:
            x[c] = 0
    x = x[EXPECTED_COLUMNS]
    return x






@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        try:
            x = _prepare_features(request.form)
            pred = model.predict(x)[0]

            prob = None
            if hasattr(model, 'predict_proba'):
                # loan_status: Approved=1 Rejected=0
                prob = float(model.predict_proba(x)[0][int(pred)])

            result = {
                'approved': bool(int(pred) == 1),
                'prob': prob,
            }
        except Exception as e:
            result = {
                'approved': None,
                'prob': None,
                'error': str(e),
            }

    return render_template('index.html', result=result)



if __name__ == '__main__':
    # http://127.0.0.1:5000
    app.run(debug=True)


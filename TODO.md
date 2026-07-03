# TODO - Fix Loan Approval UI

- [ ] Update `loan-approval-frontend/app.py` to match notebook model feature names/engineering.
- [ ] Update `_prepare_features` to build the exact columns expected by `model.feature_names_in_` (or fallback to known training columns).
- [ ] Add try/except around prediction and pass `result.error` to template.
- [ ] Update `loan-approval-frontend/templates/index.html` to collect missing fields needed by notebook model (e.g., `cibil_score` + asset values) with correct `name=` attributes.
- [ ] Update result rendering to show error when prediction fails.
- [ ] Run Flask app and manually verify form submission shows Approved/Rejected.


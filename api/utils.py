# api/utils.py
# Shared feature engineering for all endpoints

import pandas as pd


def engineer_features(application, feature_names):
    """Engineer and encode all features"""
    import pandas as pd
    import numpy as np

    # Basic engineered features
    issue_d = pd.to_datetime(application.issue_d, format='%b-%Y')
    earliest_cr_line = pd.to_datetime(application.earliest_cr_line, format='%b-%Y')
    credit_history_years = (issue_d - earliest_cr_line).days / 365
    loan_to_income = application.loan_amnt / application.annual_inc
    dti_clean = min(application.dti, 100)
    high_dti = int(dti_clean > 20)
    high_revol_util = int(application.revol_util > 80)
    has_delinquency = int(application.delinq_2yrs > 0)

    # Build base feature dictionary
    base = {
        'loan_amnt': application.loan_amnt,
        'int_rate': application.int_rate,
        'installment': application.installment,
        'annual_inc': application.annual_inc,
        'dti_clean': dti_clean,
        'delinq_2yrs': application.delinq_2yrs,
        'inq_last_6mths': application.inq_last_6mths,
        'open_acc': application.open_acc,
        'pub_rec': application.pub_rec,
        'revol_bal': application.revol_bal,
        'revol_util': application.revol_util,
        'total_acc': application.total_acc,
        'credit_history_years': credit_history_years,
        'loan_to_income': loan_to_income,
        'high_dti': high_dti,
        'high_revol_util': high_revol_util,
        'has_delinquency': has_delinquency,
    }

    # Add encoded categoricals — all zeros first
    for feat in feature_names:
        if feat not in base:
            base[feat] = 0

    # Set correct dummy to 1
    grade_col = f'grade_{application.grade}'
    purpose_col = f'purpose_{application.purpose}'
    home_col = f'home_ownership_{application.home_ownership}'

    if grade_col in base:
        base[grade_col] = 1
    if purpose_col in base:
        base[purpose_col] = 1
    if home_col in base:
        base[home_col] = 1

    # Return features in exact training order
    return np.array([[base[f] for f in feature_names]])
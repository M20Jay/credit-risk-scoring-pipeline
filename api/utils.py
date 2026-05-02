# api/utils.py
# Shared feature engineering for all endpoints

import pandas as pd


def engineer_features(application):
    """
    Engineers features from raw loan application.
    Called by both /assess and /segment endpoints.
    """
    issue_d = pd.to_datetime(
        application.issue_d, format='%b-%Y'
    )
    earliest_cr_line = pd.to_datetime(
        application.earliest_cr_line, format='%b-%Y'
    )
    credit_history_years = (
        (issue_d - earliest_cr_line).days / 365
    )
    loan_to_income = (
        application.loan_amnt / application.annual_inc
    )
    dti_clean = min(application.dti, 100)
    high_dti = int(dti_clean > 20)
    high_revol_util = int(application.revol_util > 80)
    has_delinquency = int(application.delinq_2yrs > 0)

    return {
        'credit_history_years': credit_history_years,
        'loan_to_income': loan_to_income,
        'dti_clean': dti_clean,
        'high_dti': high_dti,
        'high_revol_util': high_revol_util,
        'has_delinquency': has_delinquency
    }
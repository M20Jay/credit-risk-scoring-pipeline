# src/data/preprocessing.py
# Loads and cleans the credit risk dataset
# Called by train.py and FastAPI

import pandas as pd
import numpy as np
import yaml
import os

def load_config():
    config_path = os.path.join(
        os.getcwd(), 'configs/model.yaml'
    )
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(config):
    """Load filtered Parquet file"""
    df = pd.read_parquet(config['paths']['data'])
    print(f"Data loaded: {df.shape}")
    return df

def clean_data(df):
    """Clean raw dataset"""
    # Drop columns with more than 50% missing
    threshold =len(df) *0.5
    df = df.dropna(thresh=threshold, axis=1)

    # Cap DTI outliers
    df = df.copy()
    df['dti_clean'] = df['dti'].clip(upper=100)

# Convert date columns
    df['issue_d'] = pd.to_datetime(
        df['issue_d'], format='%b-%Y'
    )
    df['earliest_cr_line'] = pd.to_datetime(
        df['earliest_cr_line'], format='%b-%Y'
    )

    # Credit history in years
    df['credit_history_years'] = (
        (df['issue_d'] - df['earliest_cr_line']).dt.days / 365
    ).round(1)

    # Loan to income ratio
    df['loan_to_income'] = (
        df['loan_amnt'] / df['annual_inc']
    ).round(4)

    print(f"Data cleaned: {df.shape}")
    return df

def encode_categoricals(df, config):
    """Encode categorical features"""
    categorical = config['features']['categorical']

    for col in categorical:
        if col in df.columns:
            dummies = pd.get_dummies(
                df[col],
                prefix=col,
                drop_first=True
            )

            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=col, inplace=True)
    print("Categoricals encoded ✅")
    return df

def preprocess(config):
    df = load_data(config)
    df = clean_data(df)
    df = encode_categoricals(df, config)
    return df
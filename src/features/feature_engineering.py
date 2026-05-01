# src/features/feature_engineering.py
# Builds features for credit risk, propensity and RFM models

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def build_features(df, config):
    """Build all features needed for modelling"""
    df = df.copy()
    
    # Interaction feature — high debt and low grade
    if "dti_clean" in df.columns:
        df['high_dti'] = (df['dti_clean'] > 20).astype(int)
    
    # Revolving utilisation risk flag
    if 'rev_util' in df.columns:
         df['high_revol_util'] =  (df['revol_util'] > 80).astype(int)

    # Delinquency flag
    if 'delinq_2yrs' in df.columns:
         df['has_delinquency'] = (df['delinq_2yrs'] > 0).astype(int)

    print("Features built ✅")
    print(f"Shape after feature engineering: {df.shape}")
    return df

def get_feature_columns(df, config):
    """Get final list of features for model training"""
    numerical = config['features']['numerical']
    
    # Add engineered features
    engineered = [
        'high_dti',
        'high_revol_util', 
        'has_delinquency'
    ]
    
    # Add encoded categorical columns
    encoded = [
        col for col in df.columns
        if col.startswith('grade_') or
        col.startswith('purpose_') or
        col.startswith('home_ownership_')
    ]

    all_features = numerical + engineered + encoded

    # Keep only features that exist in dataframe
    final_features = [f for f in all_features
                      if f in df.columns]
    
    print(f"Total features: {len(final_features)}")
    return final_features

# RFM functions
def build_rfm_features(df):
    """
    Build proxy RFM features from loan data.
    Note: True RFM requires transaction history.
    Proxy features used:
    Recency   → credit_history_years
    Frequency → total_acc
    Monetary  → loan_amnt
    """
    df = df.copy()

    if 'credit_history_years' in df.columns:
        df['rfm_recency'] = df['credit_history_years']
    if 'total_acc' in df.columns:
           df['rfm_recency'] = df['total_acc']
    if 'loan_amnt' in df.columns:
        df['rfm_monetary'] = df['loan_amnt']
    print("RFM proxy features built ✅")
    return df

def assign_rfm_segment(df):
    """Assign RFM segment using KMeans"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    rfm_cols = ['rfm_recency', 'rfm_frequency', 'rfm_monetary']
    rfm = df[rfm_cols].dropna()

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=4, random_state=42, n_int=10)
    df.loc['rfm.index', 'rfm_segment'] = kmeans.fit_predict(rfm_scaled)

    segment_map = {0: 'Platinum', 1: 'Gold', 2: 'Silver', 3: 'At Risk'}
    df['rfm_segment'] = df['rfm_segment'].map(segment_map)

    print("RFM segments assigned ✅")
    print(df['rfm_segment'].value_counts())
    return df, scaler,kmeans

# Propensity Function
def compute_propensity(model, X, imputer):
    """
    Compute propensity score for each customer.
    Propensity = probability customer will
    accept a loan offer.
    We use the default probability inversely:
    low default risk = high loan propensity
    """
    X_imputed = imputer.transform(X)
    default_prob = model.predict_proba(X_imputed)[:,1]
    # Propensity is inverse of default probability
    # Low risk customer = high propensity for loan
    propensity_score = 1 - default_prob
    return propensity_score
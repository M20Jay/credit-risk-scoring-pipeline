# src/features/feature_engineering.py
# Builds features for credit risk, propensity and RFM models

import pandas as pd
import numpy as np

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
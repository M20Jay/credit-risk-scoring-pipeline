# src/models/train.py
# Trains credit risk model
# Applies ADASYN, calibration, threshold selection
import pandas as pd
import numpy as np
import yaml
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import ADASYN
import xgboost as xgb
from sklearn.impute import SimpleImputer
from src.data.preprocessing import load_config, preprocess
from sklearn.impute import SimpleImputer
from src.features.feature_engineering import (
    build_features, get_feature_columns,
    build_rfm_features, assign_rfm_segment
)

# Main training function
def split_data(df, features,config):
    """Three way split — train, val, test"""
    X = df[features]
    y = df[config['features']['target']]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state'],
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=config['training']['val_size'],
        random_state=config['training']['random_state'],
        stratify=y_temp
    )

    print(f"Train: {X_train.shape}")
    print(f"Val:   {X_val.shape}")
    print(f"Test:  {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def apply_adasyn(X_train, y_train):
    """Apply ADASYN on training set only"""
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)

    adasyn = ADASYN(random_state=42)
    X_resampled, y_resampled = adasyn.fit_resample(
        X_train_imputed, y_train
    )

    print(f"Before ADASYN: {y_train.value_counts().to_dict()}")
    print(f"After ADASYN:  {pd.Series(y_resampled).value_counts().to_dict()}")
    return X_resampled, y_resampled, imputer


# Model training and saving functions
def train_model(X_train, y_train, config):
    """Train XGBoost model"""
    params = config['model']['params']
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    print("Model trained ✅")
    return model

def calibrate_model(model, X_val, y_val):
    """Calibrate probabilities on validation set"""
    calibrated = CalibratedClassifierCV(
        model, cv=None, method='isotonic'
    )
    calibrated.fit(X_val, y_val)
    print("Model calibrated ✅")
    return calibrated

def find_threshold(model, X_test, y_test, config):
    """Find optimal threshold scientifically"""
    y_prob = model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(
        y_test, y_prob
    )
    target_recall = config['training']['target_recall']
    idx = next(i for i, r in enumerate(recalls)
               if r <= target_recall)
    optimal_threshold = thresholds[idx]
    print(f"Optimal threshold: {optimal_threshold.round(3)}")
    return optimal_threshold

def save_artifacts(model, threshold, imputer, config):
    """Save all artifacts in one place"""
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, config['paths']['model'])
    joblib.dump(threshold, config['paths']['threshold'])
    joblib.dump(imputer, 'models/imputer.pkl')
    print("Model saved ✅")
    print("Threshold saved ✅")
    print("Imputer saved ✅")

def compute_propensity(model, X, imputer):
    """
    Compute propensity score for each customer.
    Propensity = probability customer will
    accept a loan offer.
    Low default risk = high loan propensity.
    """
    X_imputed = imputer.transform(X)
    default_prob = model.predict_proba(X_imputed)[:, 1]
    propensity_score = 1 - default_prob
    return propensity_score

def train_pipeline():
    """Full training pipeline"""
    config = load_config()
    df = preprocess(config)
    df = build_features(df, config)
    features = get_feature_columns(df, config)

    X_train, X_val, X_test, y_train, y_val, y_test = \
        split_data(df, features, config)

    X_train, y_train, imputer = apply_adasyn(X_train, y_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    model = train_model(X_train, y_train, config)
    calibrated = calibrate_model(model, X_val, y_val)
    threshold = find_threshold(calibrated, X_test, y_test, config)
    save_artifacts(calibrated, threshold, imputer, config)
    print("Training pipeline complete ✅")
    propensity = compute_propensity(calibrated, X_test, imputer)
    print(f"Propensity scores computed ✅")
    print(f"Mean propensity: {propensity.mean().round(3)}")


if __name__ == '__main__':
    train_pipeline()
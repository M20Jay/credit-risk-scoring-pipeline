# src/models/evaluate.py
# Evaluates model and generates SHAP explanations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from src.data.preprocessing import load_config, preprocess
from src.features.feature_engineering import (
    build_features, get_feature_columns
)

def load_artifacts(config):
    """Load saved model, threshold and imputer"""
    model = joblib.load(config['paths']['model'])
    threshold =joblib.load(config['paths']['threshold'])
    imputer = joblib.load('models/imputer.pkl')
    print("Artifacts loaded ✅")
    return model, threshold, imputer


def evaluate_model(model, threshold,X_test, y_test):
    """Evaluate model performance"""
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= threshold).astype(int)
    print("=== Model Evaluation ===")
    print(f"ROC-AUC: {round(roc_auc_score(y_test, y_prob), 3)}")
    print(classification_report(y_test, y_pred, target_names =['Non-Default', 'Default']))
    return y_prob, y_pred

def plot_confusion_matrix(y_test, y_pred):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Default', 'Default'],
                yticklabels=['Non-Default', 'Default'])
    plt.title('Confusion Matrix — Credit Risk Model')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    os.makedirs('screenshots', exist_ok=True)
    plt.savefig('screenshots/confusion_matrix.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Confusion matrix saved ✅")

# SHAP functions
def generate_shap_plots(model, X_test, features):
    """Generate SHAP summary and waterfall plots"""
    base_model = model.calibrated_classifiers_[0].estimator
    
    # Use Explainer instead of TreeExplainer
    explainer = shap.Explainer(base_model, X_test[:100])
    shap_values = explainer(X_test[:100])

    # Summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values.values, X_test[:100],
        feature_names=features,
        show=False
    )
    plt.tight_layout()
    plt.savefig('screenshots/shap_summary.png',
                dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP summary plot saved ✅")

# Final evaluate pipeline function
def evaluate_pipeline():
    """Full evaluation pipeline"""
    config = load_config()
    df = preprocess(config)
    df = build_features(df, config)
    features = get_feature_columns(df,config)

    # Split to get test set
    X = df[features]
    y =df[config['features']['target']]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state'],
        stratify=y
    )

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=config['training']['test_size'],
        random_state=config['training']['random_state'],
        stratify=y
    )
    
    # Load artifacts
    model, threshold, imputer = load_artifacts(config)

    # Transform test set
    X_test_imputed = imputer.transform(X_test)

    #Evaluate
    y_prob, y_pred = evaluate_model(model, threshold, X_test_imputed, y_test)

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    # SHAP plots
    generate_shap_plots(model, X_test_imputed, features)

    print("Evaluation pipeline complete ✅")


if __name__ == '__main__':
    evaluate_pipeline()
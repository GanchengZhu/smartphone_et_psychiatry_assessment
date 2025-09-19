import json
import os
import warnings
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from depression_symptom_detection import data_reader
from depression_symptom_detection import dep_dir

# ====================
# CONSTANTS
# ====================
RANDOM_STATE = 42
METRICS_PRIORITY = ['val_recall', 'val_f1', 'val_auc', 'val_mcc', 'val_accuracy']
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'ex_fig_5')

# Classifier name mapping for display
CLASSIFIER_NAME_MAP = {
    'rf': 'RF',
    'catboost': 'CB',
    'bagging': "BAG",
    'lr': "LR"
}

# Initialize directories
os.makedirs(SAVE_DIR, exist_ok=True)


# ====================
# MODEL DEFINITIONS
# ====================
def get_classifier_map() -> Dict:
    """Return a dictionary of classifier instances."""
    return {
        'svm': SVC(kernel='rbf', C=1.0, random_state=RANDOM_STATE, probability=True),
        'lr': LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1),
        'knn': KNeighborsClassifier(n_jobs=-1),
        'nb': GaussianNB(),
        'dtree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'rf': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        'lgbm': LGBMClassifier(n_jobs=-1, random_state=RANDOM_STATE),
        'bagging': BaggingClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        'xgboost': XGBClassifier(),
        'catboost': CatBoostClassifier(eval_metric="F1", task_type='CPU',
                                       random_seed=RANDOM_STATE, verbose=False),
    }


# ====================
# UTILITY FUNCTIONS
# ====================
def load_hyperparameters(param_json_path: str) -> Dict:
    """Load hyperparameters from JSON file."""
    if not os.path.exists(param_json_path):
        warnings.warn(f"Parameter file not found: {param_json_path}")
        return {}

    try:
        with open(param_json_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        warnings.warn(f"Failed to parse parameter file: {param_json_path}")
        return {}


def setup_plot_style():
    """Configure matplotlib style settings."""
    plt.rcParams.update({
        'font.family': 'Arial',
        'mathtext.fontset': 'stix',
        'font.size': 7,
        'axes.linewidth': 1,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'figure.dpi': 600
    })


def generate_shap_plot(
        pipeline: Pipeline,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: List[str],
        save_path: str
) -> None:
    """
    Generate and save a SHAP summary plot.

    Args:
        pipeline: Trained model pipeline
        X_train: Training features (for explainer background)
        X_test: Test features to explain
        feature_names: List of feature names
        save_path: Path to save the plot
    """
    clf = pipeline.named_steps['classifier']

    # Select appropriate prediction function
    predict_fn = clf.predict_proba if hasattr(clf, "predict_proba") else clf.decision_function

    # Create explainer and calculate SHAP values
    explainer = shap.Explainer(predict_fn, X_train, feature_names=feature_names, max_evals=699)
    shap_values = explainer(X_test)

    # Handle binary classification case
    if len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
        shap_values = shap_values[:, :, 1]

    # Format feature names with mean SHAP values
    formatted_feature_names = [
        f"{name}: {abs_val:.3f}"
        for name, abs_val in zip(feature_names, np.abs(shap_values.values).mean(0))
    ]

    # Set up plot style
    setup_plot_style()

    # Create and save the plot
    fig = plt.figure(figsize=(3.3, 2.5))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=formatted_feature_names,
        plot_type="dot",
        show=False,
        max_display=15,
        color=plt.get_cmap("coolwarm")
    )

    plt.xlim((-0.125, 0.125))
    plt.tight_layout(pad=0.5)
    plt.savefig(save_path, dpi=600, bbox_inches='tight', transparent=False)
    plt.close(fig)


# ====================
# MAIN FUNCTION
# ====================
def main():
    """Main execution function for model training and evaluation."""
    label, classifier = ('overlap_depression', 'bagging')

    # Check for existing results
    clf_path = f'{dep_dir}/results/test/{label}/{RANDOM_STATE}/{classifier}/k_fold.csv'
    # Find best fold
    df = pd.read_csv(clf_path)
    valid_metrics = [col for col in METRICS_PRIORITY if col in df.columns]
    df_sorted = df.sort_values(by=valid_metrics, ascending=False)
    fold_id = df_sorted.iloc[0]['fold']
    print(f'Best fold id: {fold_id}')

    # Load and split data
    X_train, X_test, y_train, y_test, feature_names = data_reader.split_data(
        data_source=label, random_seed=RANDOM_STATE)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Initialize stratified K-fold
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    # Get classifier and create pipeline
    clf = get_classifier_map()[classifier]
    pipeline = Pipeline([('classifier', clf)])

    # Load and set hyperparameters
    param_path = f"{dep_dir}/results/train/{label}/{RANDOM_STATE}/{classifier}/best_params.json"
    param_dict = load_hyperparameters(param_path)
    pipeline.set_params(**param_dict)

    # Train on the specified fold
    for fold_idx, (train_idx, _) in enumerate(skf.split(X_train_scaled, y_train)):
        if fold_idx + 1 == fold_id:
            X_train_folder = X_train_scaled[train_idx]
            y_train_folder = y_train[train_idx]
            pipeline.fit(X_train_folder, y_train_folder)
            break

    X_test_scaled = scaler.transform(X_test)
    y_pred = pipeline.predict(X_test_scaled)
    y_prob = pipeline.predict_proba(X_test_scaled)[:, 1]

    eval_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob)
    }
    print(f"\n📊 Test Evaluation - depression ({CLASSIFIER_NAME_MAP[classifier]})")
    for metric, value in eval_metrics.items():
        print(f"{metric}: {value:.4f}")

    # Generate SHAP plot
    shap_save_path = f'{SAVE_DIR}/fig5_{label}.tiff'
    generate_shap_plot(
        pipeline, X_train_scaled, X_test_scaled, feature_names, shap_save_path
    )


if __name__ == "__main__":
    main()

# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.model_selection import train_test_split


def depression_label(row):
    """Create overlapping depression label (both PHQ9 and BDI are 1)"""
    if row['PHQ9_binary'] == 1 and row['BDI_binary'] == 1:
        return 1
    else:
        return 0


def split_data(data_source='overlap_depression', random_seed=42, test_size=0.15):
    """
    Split eye-tracking features and clinical scale scores into train/test sets

    :param data_source: Label source, options: "overlap_depression", "bdi", or "phq9"
    :param random_seed: Random seed for reproducibility
    :param test_size: Proportion of test set
    :return: X_train, X_test, y_train, y_test, feature_names
    """
    # Get current directory path
    current_dir = Path(__file__).resolve().parent

    # Define file paths using Path objects
    feature_path = current_dir / 'features' / 'depression_eyetracking_features.csv'
    scale_score_path = current_dir / 'ground_truth' / 'scale_scores.csv'

    # Load data
    features = pd.read_csv(feature_path)
    scale_scores = pd.read_csv(scale_score_path)

    # Standardize subject IDs (lowercase, strip whitespace)
    features['subj'] = features['subj'].astype(str).str.lower().str.strip()
    scale_scores['subj'] = scale_scores['subj'].astype(str).str.lower().str.strip()

    # Sort by subject ID for consistency
    features.sort_values(by='subj', inplace=True)
    scale_scores.sort_values(by='subj', inplace=True)

    # Validate data consistency
    unique_features_subj = features['subj'].nunique()
    unique_scale_subj = scale_scores['subj'].nunique()

    if unique_features_subj != unique_scale_subj:
        print(f"⚠️ Warning: Feature data has {unique_features_subj} subjects, "
              f"scale data has {unique_scale_subj} subjects")
    else:
        print(f"✅ Data matched: {unique_features_subj} subjects total")

    print(f"Feature data shape: {features.shape}")
    print(f"Scale score data shape: {scale_scores.shape}")

    # Create binary classification labels using clinical cutoffs
    # PHQ-9: >=10 indicates depression
    scale_scores['PHQ9_binary'] = scale_scores['PHQ9'].apply(lambda x: 0 if x < 10 else 1)
    # BDI: >19 indicates depression
    scale_scores['BDI_binary'] = scale_scores['BDI'].apply(lambda x: 0 if x <= 19 else 1)

    print("\n📊 Label distribution statistics:")
    print("PHQ-9 (cutoff ≥10):")
    print(scale_scores['PHQ9_binary'].value_counts().sort_index())
    print("\nBDI (cutoff >19):")
    print(scale_scores['BDI_binary'].value_counts().sort_index())

    # Create overlapping depression label (both PHQ9 and BDI positive)
    scale_scores['overlap_depression'] = scale_scores.apply(depression_label, axis=1)
    print("\nOverlapping depression label (both PHQ9 and BDI positive):")
    print(scale_scores['overlap_depression'].value_counts().sort_index())

    # Create mapping dictionaries for different label types
    label_mapping = {
        'overlap_depression': scale_scores.set_index('subj')['overlap_depression'].to_dict(),
        'phq9': scale_scores.set_index('subj')['PHQ9_binary'].to_dict(),
        'bdi': scale_scores.set_index('subj')['BDI_binary'].to_dict()
    }

    # Validate data_source parameter
    if data_source not in label_mapping:
        raise ValueError(f"data_source must be one of {list(label_mapping.keys())}")

    # Extract labels based on selected data source
    y = features['subj'].map(label_mapping[data_source]).values

    # Check for missing labels
    if np.isnan(y).any():
        missing_count = np.isnan(y).sum()
        print(f"⚠️ Warning: {missing_count} samples have missing labels")
        # Remove samples with missing labels
        valid_indices = ~np.isnan(y)
        features = features[valid_indices]
        y = y[valid_indices]

    # Extract features (excluding subject ID column)
    cleaned_feature_df = features.drop(["subj"], axis=1)
    X = cleaned_feature_df.values
    feature_names = cleaned_feature_df.columns.tolist()

    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed,
    )

    print(f"\n🎯 Data split results:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Number of features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, feature_names


def compute_feature_label_correlations(data_source='overlap_depression', random_seed=42, test_size=0.15):
    """
    Compute point-biserial correlation between each feature and the selected label.

    Returns a DataFrame sorted by absolute correlation coefficient.
    """
    # Reuse split_data to get full dataset (before split)
    current_dir = Path(__file__).resolve().parent
    feature_path = current_dir / 'features' / 'depression_eyetracking_features.csv'
    scale_score_path = current_dir / 'ground_truth' / 'scale_scores.csv'

    features = pd.read_csv(feature_path)
    scale_scores = pd.read_csv(scale_score_path)

    # Standardize subject IDs
    features['subj'] = features['subj'].astype(str).str.lower().str.strip()
    scale_scores['subj'] = scale_scores['subj'].astype(str).str.lower().str.strip()

    # Create binary labels
    scale_scores['PHQ9_binary'] = scale_scores['PHQ9'].apply(lambda x: 0 if x < 10 else 1)
    scale_scores['BDI_binary'] = scale_scores['BDI'].apply(lambda x: 0 if x <= 19 else 1)
    scale_scores['overlap_depression'] = scale_scores.apply(depression_label, axis=1)

    # Map label
    label_mapping = {
        'overlap_depression': scale_scores.set_index('subj')['overlap_depression'].to_dict(),
        'phq9': scale_scores.set_index('subj')['PHQ9_binary'].to_dict(),
        'bdi': scale_scores.set_index('subj')['BDI_binary'].to_dict()
    }

    if data_source not in label_mapping:
        raise ValueError(f"data_source must be one of {list(label_mapping.keys())}")

    features['label'] = features['subj'].map(label_mapping[data_source])

    # Remove rows with missing labels
    features = features.dropna(subset=['label'])
    y = features['label'].values.astype(int)

    # Drop non-feature columns
    feature_cols = [col for col in features.columns if col not in ['subj', 'label']]
    X_df = features[feature_cols]

    # Compute correlations
    correlations = []
    for col in feature_cols:
        corr, pval = pointbiserialr(y, X_df[col])
        correlations.append({
            'feature': col,
            'correlation': corr,
            'p_value': pval,
            '|correlation|': abs(corr)
        })

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values(by='|correlation|', ascending=False).drop(columns=['|correlation|'])
    return corr_df


if __name__ == '__main__':
    # Example usage: split data and save to files
    X_train, X_test, y_train, y_test, feature_names = split_data(
        data_source="overlap_depression", test_size=0.15)

    # # Save to files if requested
    # current_dir = Path(__file__).resolve().parent
    # save_dir = current_dir / "dl_dataset"
    # save_dir.mkdir(exist_ok=True)
    #
    # # Create training DataFrame
    # df_train = pd.DataFrame(X_train, columns=feature_names)
    # df_train["label"] = y_train
    #
    # # Create test DataFrame
    # df_test = pd.DataFrame(X_test, columns=feature_names)
    # df_test["label"] = y_test
    #
    # # Define save paths
    # train_path = save_dir / "dep_train.csv"
    # test_path = save_dir / "dep_test.csv"
    #
    # # Save to CSV
    # df_train.to_csv(train_path, index=False)
    # df_test.to_csv(test_path, index=False)
    #
    # print(f"\n💾 Data saved to:")
    # print(f"Training set: {train_path}")
    # print(f"Test set: {test_path}")
    #
    # print("\n" + "=" * 60)
    # print("🔍 Computing feature-label correlations...")
    # corr_results = compute_feature_label_correlations(data_source="overlap_depression")
    # print(corr_results.head(10))
    #
    # corr_results.to_csv(save_dir / "feature_correlations.csv", index=False)
    # print(f"\n📈 Correlation results saved to: {save_dir / 'feature_correlations.csv'}")

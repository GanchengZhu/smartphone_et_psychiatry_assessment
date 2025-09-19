# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def depression_label(row):
    if row['PHQ9_binary'] == 1 and row['BDI_binary'] == 1:
        return 1
    else:
        return 0


def split_data(data_source='overlap_depression', random_seed=42, test_size=0.2):
    """
    Split eye-tracking features and clinical scale scores into train/test sets.

    :param data_source: "overlap", "bdi", or "phq9"
    :param random_seed: random seed for reproducibility
    :param test_size: test set proportion
    :return: X_train, X_test, y_train, y_test
    """
    # Set base path relative to this script
    current_dir = Path(__file__).resolve().parent
    fv_fix_feature_path = current_dir / 'features' / 'fv_fixation_features.csv'
    fv_sac_feature_path = current_dir / 'features' / 'fv_saccade_features.csv'
    scale_score_path = current_dir / 'ground_truth' / 'scale_scores.csv'

    # Load features and ground truth
    fv_fix = pd.read_csv(fv_fix_feature_path)
    fv_sac = pd.read_csv(fv_sac_feature_path)
    scale_scores = pd.read_csv(scale_score_path)

    # Merge fixation and saccade features
    features = pd.merge(fv_fix, fv_sac, how='left', on='subj')
    features['subj'] = features['subj'].astype(str).str.lower()
    scale_scores['subj'] = scale_scores['subj'].astype(str).str.lower()

    # Sort by subject ID
    features.sort_values(by='subj', ascending=False, inplace=True)
    scale_scores.sort_values(by='subj', ascending=False, inplace=True)

    # Check data integrity
    assert np.unique(scale_scores.subj.values).size == np.unique(features.subj.values).size == 631

    print("Free viewing feature shape: ", features.shape)
    print("Scale score shape: ", scale_scores.shape)

    # Generate binary depression and anxiety labels
    scale_scores['PHQ9_binary'] = scale_scores['PHQ9'].apply(lambda x: 0 if x < 10 else 1)
    scale_scores['BDI_binary'] = scale_scores['BDI'].apply(lambda x: 0 if x <= 19 else 1)
    # scale_scores['GAD7_binary'] = scale_scores['GAD7'].apply(lambda x: 0 if x < 10 else 1)

    print("\nPHQ9 Binary Classification:")
    print(scale_scores['PHQ9_binary'].value_counts())
    print("\nBDI Binary Classification:")
    print(scale_scores['BDI_binary'].value_counts())

    scale_scores['overlap_depression'] = scale_scores.apply(depression_label, axis=1)
    print("Depression label 分类统计：")
    print(scale_scores['overlap_depression'].value_counts())

    dep_op_mapping = scale_scores.set_index('subj')['overlap_depression'].to_dict()
    dep_phq9_mapping = scale_scores.set_index('subj')['PHQ9_binary'].to_dict()
    dep_bdi_mapping = scale_scores.set_index('subj')['BDI_binary'].to_dict()
    overlap_depression = features['subj'].map(dep_op_mapping)
    phq9 = features['subj'].map(dep_phq9_mapping)
    bdi = features['subj'].map(dep_bdi_mapping)

    # Select features and labels based on data source

    if data_source == 'phq9':
        y = phq9.values
    elif data_source == 'bdi':
        y = bdi.values
    else:  # overlap
        y = overlap_depression.values
    cleaned_feature_df = features.drop(["subj",], axis=1)
    X = cleaned_feature_df.values
    feature_names = cleaned_feature_df.columns
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_names


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, _ = split_data(data_source="overlap_depression")
    print('test')

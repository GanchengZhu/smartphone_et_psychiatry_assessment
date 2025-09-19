# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import numpy as np
# load libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# eye movement feature paths
fv_fix_feature_path = '../analysis_ZW/fv_fixation_features.csv'
fv_sac_feature_path = '../analysis_ZW/fv_saccade_features.csv'

fm_fix_feature_path = '../analysis_ZW/fm_fixation_features.csv'
fm_sac_feature_path = '../analysis_ZW/fm_saccade_features.csv'

# exclude
# exclude_df = pd.read_csv('../ground_truth/to_exclude')

# load feature using pandas
fv_fix = pd.read_csv(fv_fix_feature_path)
fv_sac = pd.read_csv(fv_sac_feature_path)

fm_fix = pd.read_csv(fm_fix_feature_path)
fm_sac = pd.read_csv(fm_sac_feature_path)

# fixation feature
fix_feature_name = [i.replace('_top', '') for i in fv_fix.columns if '_top' in i]
print(fix_feature_name)

# load scale scores
scale_scores = pd.read_csv('../ground_truth/scale_scores.csv')

# merge saccade and fixation features
fv_features = pd.merge(fv_fix, fv_sac, how='left', on='subj')
fm_features = pd.merge(fm_fix, fm_sac, how='left', on='subj')
# features = pd.merge(features, fm_sac, how='left', on='subj')
# features = pd.merge(features, fm_fix, how='left', on='subj')
print(fv_features.shape)

# asserting that subject id list is unique in scale scores and eye movement feature data frame
assert np.unique(scale_scores.subj.values).size == np.unique(fv_features.subj.values).size == 631

# sort the features and scale scores
fv_features.sort_values(by='subj', ascending=False, inplace=True)
fm_features.sort_values(by='subj', ascending=False, inplace=True)
scale_scores.sort_values(by='subj', ascending=False, inplace=True)

# Generate new features
new_columns = {}

for i in fix_feature_name:
    new_columns[f'{i}_ratio'] = fv_features[f'{i}_top'] / (fv_features[f'{i}_bot'] + fv_features[f'{i}_top'])

df_new_feature_frame = pd.DataFrame(new_columns).fillna(0.5)

# Concatenate all new columns to the original DataFrame at once
features = pd.concat([fv_features, df_new_feature_frame], axis=1)

# exclude_subj = exclude_df['subj']
# features = features[~features['subj'].isin(exclude_subj)]
# scale_scores = scale_scores[~scale_scores['subj'].isin(exclude_subj)]

# printing their shape
print("Free viewing feature shape: ", features.shape)
print("Scale score shape: ", scale_scores.shape)


# Define categorization functions
def categorize_phq9(score):
    """
    Categorizes the PHQ-9 score into different levels of depression.

    Parameters:
        score (int): The score obtained from the PHQ-9 questionnaire.

    Returns:
        str: The category of depression based on the score.
    """
    if score <= 4:
        return 'No Depression'
    elif score <= 9:
        return 'Mild Depression'
    elif score <= 14:
        return 'Moderate Depression'
    elif score >= 15:
        return 'Severe Depression'


def categorize_gad7(score):
    """
    Categorizes the GAD-7 score into different levels of anxiety.

    Parameters:
        score (int): The score obtained from the GAD-7 questionnaire.

    Returns:
        str: The category of anxiety based on the score.
    """
    if score <= 4:
        return 'No Anxiety'
    elif score <= 9:
        return 'Mild Anxiety'
    elif score <= 14:
        return 'Moderate Anxiety'
    else:
        return 'Severe Anxiety'


def categorize_sas(score):
    """
    Categorizes the SAS score into different levels of anxiety.

    Parameters:
        score (int): The score obtained from the SAS questionnaire.

    Returns:
        str: The category of anxiety based on the score.
    """
    if score <= 40:
        return 'Normal'
    elif score > 41:
        return 'Anxiety'


def categorize_bdi(score):
    """
    Categorizes the BDI score into different levels of depression.

    Parameters:
        score (int): The score obtained from the BDI questionnaire.

    Returns:
        str: The category of depression based on the score.
    """
    if score <= 9:
        return 'No Depression'
    elif score <= 19:
        return 'Mild Depression'
    elif score <= 29:
        return 'Moderate Depression'
    else:
        return 'Severe Depression'


# Apply categorization functions to each scale score column
scale_scores['PHQ9_category'] = scale_scores['PHQ9'].apply(categorize_phq9)
scale_scores['GAD7_category'] = scale_scores['GAD7'].apply(categorize_gad7)
# scale_scores['SAS_category'] = scale_scores['SAS'].apply(categorize_sas)
scale_scores['BDI_category'] = scale_scores['BDI'].apply(categorize_bdi)

# convert into categoryiesfor multi-class classification
label_encoder = LabelEncoder()
scale_scores['PHQ9_multi_class'] = label_encoder.fit_transform(scale_scores['PHQ9_category'])
scale_scores['GAD7_multi_class'] = label_encoder.fit_transform(scale_scores['GAD7_category'])
scale_scores['BDI_multi_class'] = label_encoder.fit_transform(scale_scores['BDI_category'])

# Count the number of people in each category for each scale
phq9_counts = scale_scores['PHQ9_category'].value_counts()
gad7_counts = scale_scores['GAD7_category'].value_counts()
# sas_counts = scale_scores['SAS_category'].value_counts()
bdi_counts = scale_scores['BDI_category'].value_counts()

# Output the categorization statistics for each scale
print("PHQ9 Categorization Statistics:")
print(phq9_counts)
print("\nGAD7 Categorization Statistics:")
print(gad7_counts)
# print("\nSAS Categorization Statistics:")
# print(sas_counts)
print("\nBDI Categorization Statistics:")
print(bdi_counts)



# Add binary classification for depression
scale_scores['PHQ9_binary'] = scale_scores['PHQ9'].apply(
    lambda x: 0 if x < 10 else 1)
scale_scores['BDI_binary'] = scale_scores['BDI_category'].apply(
    lambda x: 0 if x in ['No Depression', 'Mild Depression'] else 1)
# Add binary classification for anxiety
scale_scores['GAD7_binary'] = scale_scores['GAD7'].apply(
    lambda x: 0 if x < 10 else 1)

# Count the number of people in each category for each scale (with binary classification)
phq9_counts_binary = scale_scores['PHQ9_binary'].value_counts()
gad7_counts_binary = scale_scores['GAD7_binary'].value_counts()
bdi_counts_binary = scale_scores['BDI_binary'].value_counts()

# Output the binary classification statistics for each scale
print("\nPHQ9 Binary Classification:")
print(phq9_counts_binary)
print("\nGAD7 Binary Classification:")
print(gad7_counts_binary)
print("\nBDI Binary Classification:")
print(bdi_counts_binary)

#
def depression_label(row):
    # if row['PHQ9_binary'] == 1 and row['BDI_binary'] == 1:
    #     return 1
    # else:
    #     return 0
    if row['PHQ9_binary'] == 0 and row['BDI_binary'] == 0:
        return 0
    elif row['PHQ9_binary'] == 1 and row['BDI_binary'] == 1:
        return 1
    else:
        return -1

# 创建新的 Depression label 列
scale_scores['overlap_depression'] = scale_scores.apply(depression_label, axis=1)
# 打印 Depression label 分类统计
print("Depression label 分类统计：")
print(scale_scores['overlap_depression'].value_counts())

features = features[scale_scores['overlap_depression'] != -1]

X = features.drop(["subj"], axis=1).values
y = scale_scores[scale_scores['overlap_depression'] != -1].overlap_depression.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

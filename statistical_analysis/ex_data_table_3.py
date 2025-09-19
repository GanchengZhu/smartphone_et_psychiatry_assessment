import os.path
from pathlib import Path

import numpy as np
import pandas as pd

from depression_symptom_detection import dep_dir
from utlis import get_stats_formatted


def depression_label(row):
    if row['PHQ9_binary'] == 1 and row['BDI_binary'] == 1:
        return 1
    else:
        return 0


dep_folder = Path(dep_dir)

fv_fix_feature_path = dep_folder / 'features' / 'fv_fixation_features.csv'
fv_sac_feature_path = dep_folder / 'features' / 'fv_saccade_features.csv'
scale_score_path = dep_folder / 'ground_truth' / 'scale_scores.csv'

# Load features and ground truth
fv_fix = pd.read_csv(fv_fix_feature_path)
fv_sac = pd.read_csv(fv_sac_feature_path)
scale_scores = pd.read_csv(scale_score_path)

# Merge fixation and saccade features
features_df = pd.merge(fv_fix, fv_sac, how='left', on='subj')
features_df['subj'] = features_df['subj'].astype(str).str.lower()
scale_scores['subj'] = scale_scores['subj'].astype(str).str.lower()

# Sort by subject ID
features_df.sort_values(by='subj', ascending=False, inplace=True)
scale_scores.sort_values(by='subj', ascending=False, inplace=True)

# Check data integrity
assert np.unique(scale_scores.subj.values).size == np.unique(features_df.subj.values).size == 631

print("Free viewing feature shape: ", features_df.shape)
print("Scale score shape: ", scale_scores.shape)

# Generate binary depression and anxiety labels
scale_scores['PHQ9_binary'] = scale_scores['PHQ9'].apply(lambda x: 0 if x < 10 else 1)
scale_scores['BDI_binary'] = scale_scores['BDI'].apply(lambda x: 0 if x <= 19 else 1)

scale_scores['overlap_depression'] = scale_scores.apply(depression_label, axis=1)
dep_op_mapping = scale_scores.set_index('subj')['overlap_depression'].to_dict()
features_df.loc[:, 'label'] = features_df['subj'].map(dep_op_mapping)
features_df.drop(["subj", ], axis=1, inplace=True)
# 创建结果DataFrame
results = pd.DataFrame(columns=['Feature', 'Non-symptomatic Mean±(95% CI)', 'Symptomatic Mean±(95% CI)',
                                't-value', 'p-value', "Cohen's d", "d 95% CI"])

# 对每个特征进行统计分析
feature_names = features_df.columns.drop('label')

for feature_name in feature_names:
    stats = get_stats_formatted(features_df, feature_name)
    results.loc[len(results)] = [feature_name] + list(stats)

table_save_dir = os.path.join(os.path.dirname(__file__), 'ex_table_3')
os.makedirs(table_save_dir, exist_ok=True)
# 显示结果
results.to_excel(f'{table_save_dir}/depression_stats_ttest.xlsx', index=False)

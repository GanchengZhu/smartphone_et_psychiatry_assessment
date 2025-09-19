# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from depression_symptom_detection import data_reader

# 假设 scale_scores 中有 'subj' 和 'label' 列，label=1 为症状组，0 为对照组
# 合并标签到 features
X_train, X_test, y_train, y_test, feature_names = data_reader.split_data(
    data_source='overlap',
)
# 合并训练和测试数据
X_all = np.vstack([X_train, X_test])  # shape: (N, D)
y_all = np.concatenate([y_train, y_test])  # shape: (N,)

# 转为 DataFrame 方便操作
df_all = pd.DataFrame(X_all, columns=feature_names)
df_all['label'] = y_all

# 用于存储结果
results = []

for feature in feature_names:
    group0 = df_all[df_all['label'] == 0][feature].dropna()
    group1 = df_all[df_all['label'] == 1][feature].dropna()

    if group0.empty or group1.empty:
        continue

    t_stat, p_value = ttest_ind(group0, group1, equal_var=False)
    results.append({
        'feature': feature,
        't_stat': t_stat,
        'p_value': p_value
    })

# 转为 DataFrame
t_df = pd.DataFrame(results).sort_values(by='p_value')

# 可选：多重比较校正（FDR）
pvals = t_df['p_value'].values
rejected, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
t_df['p_fdr'] = pvals_corrected
t_df['significant'] = rejected

# 打印显著的特征（可选）
# print(t_df[t_df['significant']])
# 保存完整结果到 CSV
t_df.to_csv('t_test_results.csv', index=False)

# （可选）只保存显著的特征
t_df[t_df['significant']].to_csv('t_test_significant_results.csv', index=False)
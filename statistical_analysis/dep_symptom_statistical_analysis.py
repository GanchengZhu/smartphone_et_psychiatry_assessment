# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, pearsonr

# === 读取数据 ===
df = pd.read_excel('depression_symptom_all_scale_score.xlsx')  # 包含年龄、性别、PHQ9、BDI
binary_df = pd.read_excel('depression_symptom_label.xlsx')  # 包含标签 overlap_depression

# 合并数据（推荐）
df = pd.merge(df, binary_df, on='subj')
label = df['overlap_depression']
age = df['age']
gender = df['gender']
phq9_scores = df['PHQ9']
bdi_scores = df['BDI']

# === 分组 ===
non_symptom_age = age[label == 0]
symptom_age = age[label == 1]

non_symptom_gender = gender[label == 0]
symptom_gender = gender[label == 1]

non_symptom_phq9 = phq9_scores[label == 0]
symptom_phq9 = phq9_scores[label == 1]

non_symptom_bdi = bdi_scores[label == 0]
symptom_bdi = bdi_scores[label == 1]

# === 年龄统计分析 ===
print("年龄统计：")
print("Non-symptom group：", len(non_symptom_age))
print("Symptom group：", len(symptom_age))
mean_age_non = np.nanmean(non_symptom_age)
std_age_non = np.nanstd(non_symptom_age)
mean_age_sym = np.nanmean(symptom_age)
std_age_sym = np.nanstd(symptom_age)
print(f"Non-symptom: {mean_age_non:.2f} ± {std_age_non:.2f}")
print(f"Symptom: {mean_age_sym:.2f} ± {std_age_sym:.2f}")

# t 检验 + Cohen's d
t_stat_age, p_val_age = ttest_ind(non_symptom_age, symptom_age, equal_var=False)
pooled_std_age = np.sqrt(((len(non_symptom_age) - 1) * std_age_non**2 + (len(symptom_age) - 1) * std_age_sym**2) /
                         (len(non_symptom_age) + len(symptom_age) - 2))
cohen_d_age = (mean_age_sym - mean_age_non) / pooled_std_age

# === 性别卡方检验 ===
male_non = sum(non_symptom_gender == '男')
female_non = sum(non_symptom_gender == '女')
male_sym = sum(symptom_gender == '男')
female_sym = sum(symptom_gender == '女')

table = np.array([[male_non, male_sym], [female_non, female_sym]])
chi2, p_gender, dof, expected = chi2_contingency(table)
# === 性别效果量：Phi coefficient (rφ) ===
n_total = np.sum(table)
phi = np.sqrt(chi2 / n_total)

# === PHQ9 统计 + Cohen's d ===
mean_phq9_non = np.nanmean(non_symptom_phq9)
std_phq9_non = np.nanstd(non_symptom_phq9)
mean_phq9_sym = np.nanmean(symptom_phq9)
std_phq9_sym = np.nanstd(symptom_phq9)

t_stat_phq9, p_val_phq9 = ttest_ind(non_symptom_phq9, symptom_phq9, equal_var=False)
pooled_std_phq9 = np.sqrt(((len(non_symptom_phq9) - 1) * std_phq9_non**2 + (len(symptom_phq9) - 1) * std_phq9_sym**2) /
                          (len(non_symptom_phq9) + len(symptom_phq9) - 2))
cohen_d_phq9 = (mean_phq9_sym - mean_phq9_non) / pooled_std_phq9

# === BDI 统计 + Cohen's d ===
mean_bdi_non = np.nanmean(non_symptom_bdi)
std_bdi_non = np.nanstd(non_symptom_bdi)
mean_bdi_sym = np.nanmean(symptom_bdi)
std_bdi_sym = np.nanstd(symptom_bdi)

t_stat_bdi, p_val_bdi = ttest_ind(non_symptom_bdi, symptom_bdi, equal_var=False)
pooled_std_bdi = np.sqrt(((len(non_symptom_bdi) - 1) * std_bdi_non**2 + (len(symptom_bdi) - 1) * std_bdi_sym**2) /
                         (len(non_symptom_bdi) + len(symptom_bdi) - 2))
cohen_d_bdi = (mean_bdi_sym - mean_bdi_non) / pooled_std_bdi

# === PHQ9 与 BDI 的相关性 ===
r_val, p_corr = pearsonr(phq9_scores, bdi_scores)

# === 打印最终表格数据 ===
df_age = len(non_symptom_age) + len(symptom_age) - 2
df_phq9 = len(non_symptom_phq9) + len(symptom_phq9) - 2
df_bdi = len(non_symptom_bdi) + len(symptom_bdi) - 2

print("\n=== Table 2 汇总结果（含自由度） ===")
print(f"Age: Sym={mean_age_sym:.2f} ± {std_age_sym:.2f}, Non={mean_age_non:.2f} ± {std_age_non:.2f}, "
      f"t({df_age})={t_stat_age:.2f}, p={p_val_age:.4f}, d={cohen_d_age:.2f}")

print(f"Gender: Male={male_sym}/{male_non}, Female={female_sym}/{female_non}, "
      f"χ²({dof})={chi2:.2f}, p={p_gender:.4f}, phi={phi}")

print(f"PHQ9: Sym={mean_phq9_sym:.2f} ± {std_phq9_sym:.2f}, Non={mean_phq9_non:.2f} ± {std_phq9_non:.2f}, "
      f"t({df_phq9})={t_stat_phq9:.2f}, p={p_val_phq9:.4f}, d={cohen_d_phq9:.2f}")

print(f"BDI: Sym={mean_bdi_sym:.2f} ± {std_bdi_sym:.2f}, Non={mean_bdi_non:.2f} ± {std_bdi_non:.2f}, "
      f"t({df_bdi})={t_stat_bdi:.2f}, p={p_val_bdi:.4f}, d={cohen_d_bdi:.2f}")

print(f"PHQ9 vs BDI: r={r_val:.2f}, p={p_corr:.4f}")
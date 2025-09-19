# -*- coding: utf-8 -*-
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ttest_ind

from schizophrenia_detection import sz_dir

# Load SZ metadata
sz_data_path = f"{sz_dir}/meta_data/meta_data_sz.xlsx"
sz_batch_0 = pd.read_excel(sz_data_path, sheet_name="batch_0", na_values='NAN')
sz_batch_1 = pd.read_excel(sz_data_path, sheet_name="batch_1")

# Concatenate SZ data
sz_age = np.concatenate((sz_batch_0['age'], sz_batch_1['age']))
sz_gender = np.concatenate(
    (sz_batch_0['gender'].astype(str).str.strip().values, sz_batch_1['gender'].astype(str).str.strip().values))
sz_saps = np.concatenate((sz_batch_0['SAPS'], sz_batch_1['SAPS']))
sz_bprs = np.concatenate((sz_batch_0['BPRS'], sz_batch_1['BPRS']))
sz_sans = np.concatenate((sz_batch_0['SANS'], sz_batch_1['SANS']))
sz_panns = np.concatenate((sz_batch_0['PANNS'], sz_batch_1['PANNS']))

# Print SZ stats
print("=== Schizophrenia (SZ) Group ===")
print(f"Age: {np.nanmean(sz_age):.2f} ± {np.nanstd(sz_age):.2f} (range: {np.nanmin(sz_age)} to {np.nanmax(sz_age)})")
print(f"Gender (M:F): {sum(sz_gender == 'male')} : {sum(sz_gender == 'female')}")
print(f"SAPS: {np.nanmean(sz_saps):.2f} ± {np.nanstd(sz_saps):.2f}")
print(f"BPRS: {np.nanmean(sz_bprs):.2f} ± {np.nanstd(sz_bprs):.2f}")
print(f"SANS: {np.nanmean(sz_sans):.2f} ± {np.nanstd(sz_sans):.2f}")
print(f"PANSS: {np.nanmean(sz_panns):.2f} ± {np.nanstd(sz_panns):.2f}")

# Load TC metadata
tc_data_path = f"{sz_dir}/meta_data/meta_data_tc.xlsx"
tc_batch_0 = pd.read_excel(tc_data_path, sheet_name="batch_0")
tc_batch_1 = pd.read_excel(tc_data_path, sheet_name="batch_1")

# Concatenate TC data
tc_age = np.concatenate((tc_batch_0['age'], tc_batch_1['age']))
tc_gender = np.concatenate((tc_batch_0['gender'], tc_batch_1['gender']))
tc_spq = np.concatenate((tc_batch_0['score'], tc_batch_1['score']))

# Print TC stats
print("\n=== Typical Control (TC) Group ===")
print(f"Age: {np.nanmean(tc_age):.2f} ± {np.nanstd(tc_age):.2f} (range: {np.nanmin(tc_age)} to {np.nanmax(tc_age)})")
print(f"Gender (M:F): {sum(tc_gender == 'male')} : {sum(tc_gender == 'female')}")
print(f"SPQ: {np.nanmean(tc_spq):.2f} ± {np.nanstd(tc_spq):.2f}")

# Chi-square test on gender
contingency_table = np.array([
    [sum(sz_gender == 'male'), sum(tc_gender == 'male')],
    [sum(sz_gender == 'female'), sum(tc_gender == 'female')]
])
chi2, p_chi2, dof, expected = chi2_contingency(contingency_table)

# Calculate Phi coefficient (effect size for 2x2 table)
n = contingency_table.sum()  # Total sample size
phi = np.sqrt(chi2 / n)

print("\n=== Statistical Tests ===")
print("1. Gender Distribution (Chi-square Test):")
print(f"   - χ² = {chi2:.4f}, df = {dof}, p = {p_chi2:.4f}")
print(f"   - Phi coefficient (effect size) = {phi:.4f}")
print("   Interpretation:")
print("   - Phi ≈ 0.1: Small effect")
print("   - Phi ≈ 0.3: Medium effect")
print("   - Phi ≈ 0.5: Large effect")

# Welch's t-test on age
sz_age_clean = sz_age[~np.isnan(sz_age)]
t_stat, p_val = ttest_ind(sz_age_clean, tc_age, equal_var=False)


# Cohen's d for age
def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.std(x, ddof=1) ** 2 + (ny - 1) * np.std(y, ddof=1) ** 2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std


d = cohen_d(sz_age_clean, tc_age)


# Welch's degrees of freedom calculation
def welch_df(x, y):
    # s1 = np.var(x, ddof=1)
    # s2 = np.var(y, ddof=1)
    # n1 = len(x)
    # n2 = len(y)
    # numerator = (s1 / n1 + s2 / n2) ** 2
    # denominator = (s1**2 / (n1**2 * (n1 - 1))) + (s2**2 / (n2**2 * (n2 - 1)))
    # return numerator / denominator
    n1 = len(x)
    n2 = len(y)
    return n1 + n2 - 2


df_welch = welch_df(sz_age_clean, tc_age)

print("\n2. Age Difference (Welch's t-test):")
print(f"   - t = {t_stat:.4f}, df = {df_welch:.2f}, p = {p_val:.4f}")
print(f"   - t = {t_stat:.4f}, p = {p_val:.4f}")
print(f"   - Cohen's d (effect size) = {d:.4f}")
print("   Interpretation:")
print("   - d ≈ 0.2: Small effect")
print("   - d ≈ 0.5: Medium effect")
print("   - d ≈ 0.8: Large effect")

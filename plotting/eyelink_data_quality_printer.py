# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# =======================
# 读取数据
# =======================
sz_phone_df = pd.read_csv(os.path.join("data_quality", "sz_phone_fs_data_quality.csv"))

# 分组
group0 = sz_phone_df[sz_phone_df['sz'] == 0]
group1 = sz_phone_df[sz_phone_df['sz'] == 1]

# 提取指标
acc_0 = group0['acc'].dropna().values
acc_1 = group1['acc'].dropna().values

pre_0 = group0['pre'].dropna().values
pre_1 = group1['pre'].dropna().values

# =======================
# 计算函数
# =======================
def pooled_sd_weighted(g1, g2):
    n1, n2 = len(g1), len(g2)
    s1 = np.var(g1, ddof=1)
    s2 = np.var(g2, ddof=1)
    pooled_var = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    return np.sqrt(pooled_var)

def cohen_d_pooled(g1, g2):
    diff = np.mean(g1) - np.mean(g2)
    sd = pooled_sd_weighted(g1, g2)
    return diff / sd

def hedges_g(d, n1, n2):
    df = n1 + n2 - 2
    J = 1 - 3 / (4 * df - 1)
    return d * J

def cohen_d_bootstrap_ci(g1, g2, n_boot=5000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed)
    n1, n2 = len(g1), len(g2)
    boot_d = []
    for _ in range(n_boot):
        s1 = rng.choice(g1, size=n1, replace=True)
        s2 = rng.choice(g2, size=n2, replace=True)
        pooled = pooled_sd_weighted(s1, s2)
        boot_d.append((np.mean(s1) - np.mean(s2)) / pooled)
    ci_low, ci_high = np.percentile(boot_d, [100 * alpha/2, 100 * (1 - alpha/2)])
    return np.mean(boot_d), (ci_low, ci_high)

# =======================
# t检验
# =======================
t_stat_acc, p_val_acc = ttest_ind(acc_0, acc_1, equal_var=True)
t_stat_pre, p_val_pre = ttest_ind(pre_0, pre_1, equal_var=True)

# =======================
# Cohen's d / Hedges' g / CI
# =======================
d_acc = cohen_d_pooled(acc_0, acc_1)
g_acc = hedges_g(d_acc, len(acc_0), len(acc_1))
d_acc_boot_mean, (d_acc_ci_low, d_acc_ci_high) = cohen_d_bootstrap_ci(acc_0, acc_1)

d_pre = cohen_d_pooled(pre_0, pre_1)
g_pre = hedges_g(d_pre, len(pre_0), len(pre_1))
d_pre_boot_mean, (d_pre_ci_low, d_pre_ci_high) = cohen_d_bootstrap_ci(pre_0, pre_1)

# =======================
# 打印结果
# =======================
def print_stats(name, g0, g1, t_stat, p_val, d, g, ci_low, ci_high):
    n0, n1 = len(g0), len(g1)
    mean0, std0 = np.mean(g0), np.std(g0, ddof=1)
    mean1, std1 = np.mean(g1), np.std(g1, ddof=1)
    df = n0 + n1 - 2
    print(f"{name}:")
    print(f"  Group 0 (Healthy):  n={n0}, M={mean0:.2f}, SD={std0:.2f}")
    print(f"  Group 1 (SZ):       n={n1}, M={mean1:.2f}, SD={std1:.2f}")
    print(f"  t({df}) = {t_stat:.2f}, p = {p_val:.4f}")
    print(f"  Cohen's d = {d:.3f}, Hedges' g = {g:.3f}")
    print(f"  95% CI for d (bootstrap) = [{ci_low:.3f}, {ci_high:.3f}]\n")

print_stats("ACC", acc_0, acc_1, t_stat_acc, p_val_acc, d_acc, g_acc, d_acc_ci_low, d_acc_ci_high)
print_stats("PRE", pre_0, pre_1, t_stat_pre, p_val_pre, d_pre, g_pre, d_pre_ci_low, d_pre_ci_high)

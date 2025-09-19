# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import numpy as np
from scipy.stats import ttest_ind


def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    """计算均值的bootstrap置信区间"""
    if len(data) == 0:
        return np.nan, np.nan

    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrapped_means.append(np.mean(sample))

    lower = np.percentile(bootstrapped_means, (100 - ci) / 2)
    upper = np.percentile(bootstrapped_means, 100 - (100 - ci) / 2)
    return lower, upper


# 修改get_stats_formatted函数中的CI显示格式
def get_stats_formatted(data, feature):
    group0 = data[data['label'] == 0][feature].dropna()
    group1 = data[data['label'] == 1][feature].dropna()

    # 计算均值和标准差
    mean0, mean1 = np.mean(group0), np.mean(group1)
    std0, std1 = np.std(group0, ddof=1), np.std(group1, ddof=1)

    # 计算均值的置信区间
    ci_lower0, ci_upper0 = bootstrap_ci(group0)
    ci_lower1, ci_upper1 = bootstrap_ci(group1)

    # 独立样本t检验
    t_val, p_val = ttest_ind(group0, group1, equal_var=False)

    # 计算效应量 (Cohen's d)
    n0, n1 = len(group0), len(group1)
    pooled_std = np.sqrt(((n0 - 1) * std0 ** 2 + (n1 - 1) * std1 ** 2) / (n0 + n1 - 2))
    cohen_d = (mean0 - mean1) / pooled_std

    # 计算效应量的置信区间
    def cohen_d_ci(x, y, ci=0.95, n_bootstrap=1000):
        ds = []
        for _ in range(n_bootstrap):
            sample_x = np.random.choice(x, size=len(x), replace=True)
            sample_y = np.random.choice(y, size=len(y), replace=True)
            mean_x, mean_y = np.mean(sample_x), np.mean(sample_y)
            std_x, std_y = np.std(sample_x, ddof=1), np.std(sample_y, ddof=1)
            n_x, n_y = len(sample_x), len(sample_y)
            pooled_std = np.sqrt(((n_x - 1) * std_x ** 2 + (n_y - 1) * std_y ** 2) / (n_x + n_y - 2))
            d = (mean_x - mean_y) / pooled_std
            ds.append(d)
        lower = np.percentile(ds, (1 - ci) / 2 * 100)
        upper = np.percentile(ds, (1 + ci) / 2 * 100)
        return lower, upper

    d_lower, d_upper = cohen_d_ci(group0, group1)

    def fmt_p(p):
        if p < 0.001:
            return "< 0.001 ***"
        elif p < 0.01:
            return f"{p:.3f} **"
        elif p < 0.05:
            return f"{p:.3f} *"
        elif p < 0.1:
            return f"{p:.3f} †"
        else:
            return f"{p:.3f}"

    return (
        f"{mean0:.3f}±({ci_lower0:.3f},{ci_upper0:.3f})",
        f"{mean1:.3f}±({ci_lower1:.3f},{ci_upper1:.3f})",
        f"{t_val:.2f}",
        fmt_p(p_val),
        f"{cohen_d:.3f}",
        f"({d_lower:.3f},{d_upper:.3f})",
    )

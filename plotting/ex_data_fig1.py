import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from scipy.stats import ttest_ind

from schizophrenia_detection import sz_dir

plt.rcParams.update({
    'font.size': 7,
    'axes.titlesize': 7,
    'axes.labelsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'font.sans-serif': 'Arial',
    'axes.linewidth': 0.5,
    'lines.linewidth': 0.8,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'savefig.dpi': 600,
    'figure.dpi': 600
})


def box_plotting(ax, df, metric, title, group_labels, ylim=None):
    palette = {'0': '#1f77b4', '1': '#d62728'}  # Nature蓝红
    df = df.copy()

    sns.boxplot(
        data=df, x='label', y=metric, ax=ax, palette=palette,
        showfliers=True, width=0.6, linewidth=0.8, saturation=1,
        boxprops=dict(alpha=0.7)
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(group_labels[::-1], fontsize=6)
    ax.set_ylabel(f'{metric.replace("mean_", "").capitalize()} (°)', fontsize=6)
    ax.set_xlabel(title, fontsize=7)
    if ylim is not None:
        ax.set_ylim(ylim)
    sns.despine(ax=ax, trim=True)


cal_res = pd.read_csv(f'{os.path.dirname(__file__)}/data_quality/sz_eyelink_accuracy.csv')

_meta = pd.concat([
    pd.read_excel(f'{sz_dir}/meta_data/meta_data_release.xlsx', sheet_name=f'batch_{i}')
    for i in [0, 1]
])

group_mapping = _meta.set_index('id')['sz'].to_dict()
cal_res['sz'] = cal_res['id'].map(group_mapping)
cal_res['task'] = pd.Categorical(cal_res['task_str'])
cal_res['group'] = cal_res['sz'].map({0: 'HC', 1: 'SZ'})

# palette = ["#3C5488FF", "#00A087FF"]
palette = ['#1f77b4', '#d62728']
group_labels = ['HC', 'SZ']
task_order = ['fx', 'fv', 'ps']
task_names = {'fx': 'FS', 'fv': 'FV', 'ps': 'SP'}

fig_width, fig_height = 7.09, 4.8
fig = plt.figure(figsize=(fig_width, fig_height))
gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1.6], hspace=0.3, wspace=0.4)

# 图 a: Distribution
for g, group in enumerate(group_labels):
    for i, task in enumerate(task_order):
        ax = fig.add_subplot(gs[g, i])
        subset = cal_res[(cal_res['task_str'] == task) & (cal_res['group'] == group)]
        mean_val = np.nanmean(subset['mean_accuracy'])

        ax.hist(subset['mean_accuracy'], bins=np.arange(0, 5.1, 0.2),
                color=palette[g], alpha=0.75, edgecolor='black', linewidth=0.3)

        ax.set_xlim([0, 5])
        ax.set_ylim([0, 80])
        ax.set_xticks([0, 3.0])
        ax.set_yticks([0, 40, 80])
        ax.tick_params(axis='both', length=2, pad=1)

        if i == 0:
            ax.set_ylabel(f'{group}', labelpad=2)
        else:
            ax.set_ylabel('')
        ax.set_xlabel('Accuracy (°)', labelpad=1)

        ax.text(0.1, 72, task_names[task], fontsize=7, weight='bold')
        ax.text(3.0, 15, f'μ = {mean_val:.2f}°', fontsize=6)

# 图 b: 箱型图
axv = fig.add_subplot(gs[:, 3])
summary_df = cal_res.copy()
summary_df['label'] = summary_df['sz'].astype(str)
summary_df['task_str'] = 'All tasks'

box_plotting(axv, summary_df, 'mean_accuracy', title='All tasks', group_labels=group_labels)

# 显著性标记
hc_vals = summary_df[summary_df['label'] == '0']['mean_accuracy']
sz_vals = summary_df[summary_df['label'] == '1']['mean_accuracy']
t_stat, p_val = ttest_ind(hc_vals, sz_vals, nan_policy='omit')




def pooled_sd_weighted(g1, g2):
    """
    加权 pooled SD，用于 Cohen's d
    """
    n1, n2 = len(g1), len(g2)
    s1 = np.var(g1, ddof=1)
    s2 = np.var(g2, ddof=1)
    pooled_var = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    return np.sqrt(pooled_var)


def cohen_d_pooled(g1, g2):
    """
    计算 Cohen's d
    """
    diff = np.mean(g1) - np.mean(g2)
    sd = pooled_sd_weighted(g1, g2)
    return diff / sd


def hedges_g(d, n1, n2):
    """
    小样本偏差校正：Hedges' g
    """
    df = n1 + n2 - 2
    J = 1 - 3 / (4 * df - 1)
    return d * J


def cohen_d_bootstrap_ci(g1, g2, n_boot=5000, seed=42, alpha=0.05):
    """
    用 bootstrap 直接计算 Cohen's d 的 95% CI
    """
    rng = np.random.default_rng(seed)
    n1, n2 = len(g1), len(g2)
    boot_d = []
    for _ in range(n_boot):
        s1 = rng.choice(g1, size=n1, replace=True)
        s2 = rng.choice(g2, size=n2, replace=True)
        pooled = pooled_sd_weighted(s1, s2)
        boot_d.append((np.mean(s1) - np.mean(s2)) / pooled)
    ci_low, ci_high = np.percentile(boot_d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return np.mean(boot_d), (ci_low, ci_high)


# =======================
# 使用示例
# hc_vals 和 sz_vals 是你的两组数据
# =======================

# 点估计
d_point = cohen_d_pooled(hc_vals, sz_vals)
g_point = hedges_g(d_point, len(hc_vals), len(sz_vals))

# 95% CI (bootstrap)
d_boot_mean, (ci_low, ci_high) = cohen_d_bootstrap_ci(hc_vals, sz_vals)


n1, n2 = len(hc_vals), len(sz_vals)
df = n1 + n2 - 2
print(f"Cohen's d = {d_point:.3f}")
print(f"Hedges' g = {g_point:.3f}")
print(f"95% CI for d (bootstrap) = [{ci_low:.3f}, {ci_high:.3f}]")

y_max = max(hc_vals.max(), sz_vals.max()) + 0.3
axv.plot([0, 0, 1, 1], [y_max, y_max + 0.1, y_max + 0.1, y_max],
         lw=0.6, c='black')

sig = r'*** $p < 0.001$' if p_val < 0.001 else \
    r'** $p < 0.01$' if p_val < 0.01 else \
        r'* $p < 0.05$' if p_val < 0.05 else 'ns'

axv.text(0.5, y_max + 0.15, sig, ha='center', va='bottom', fontsize=7)

fig.text(0.07, 0.90, 'a', weight='bold', fontsize=8)
fig.text(0.65, 0.90, 'b', weight='bold', fontsize=8)

plt.tight_layout()

out_dir = f'{os.path.dirname(__file__)}/ex_fig_1'
os.makedirs(out_dir, exist_ok=True)
plt.savefig(f'{out_dir}/ex_fig_1.tiff', dpi=600, bbox_inches='tight')

hc_mean, hc_std = hc_vals.mean(), hc_vals.std()
sz_mean, sz_std = sz_vals.mean(), sz_vals.std()

print('\n\n================= Publication-ready Text =================\n')
print(
    f'To evaluate group-level differences in gaze behavior, we compared the healthy control (HC) group with individuals diagnosed with schizophrenia (SZ).')
print(
    f'On average, the SZ group demonstrated lower gaze accuracy, with a mean angular error of {sz_mean:.2f}° (SD = {sz_std:.2f}°), compared to {hc_mean:.2f}° (SD = {hc_std:.2f}°) in the HC group.')
print(
    f'An independent two-sample t-test revealed that these group differences were statistically significant (t({df}) = {t_stat:.2f}, p = {p_val:.3g}, Cohen’s d = {d_point:.2f}, 95% CI = [{ci_low:.2f}, {ci_high:.2f}]).\n')
print('=========================================================\n\n')

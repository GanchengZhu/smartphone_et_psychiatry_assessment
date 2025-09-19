import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from schizophrenia_detection import sz_dir

# ===== 设置Nature子刊风格 =====
def set_nature_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 6,
        'axes.titlesize': 6,
        'axes.labelsize': 6,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'figure.figsize': (6.7, 2.2),  # Nature双栏图宽度: 6.7英寸，高度自定
        'figure.dpi': 600,
        'axes.linewidth': 1,
        'grid.linewidth': 1,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'legend.frameon': False,
        'legend.title_fontsize': 6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
        'axes.edgecolor': '.15',
        'grid.color': '.9',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

set_nature_style()

# ===== 读取数据 =====
eyelink_path = f'{sz_dir}/features/data_eyelink'
phone_path = f'{sz_dir}/features/data_phone'

def load_and_merge_batches(base_path):
    df0 = pd.read_excel(f"{base_path}/batch_0.xlsx")
    df1 = pd.read_excel(f"{base_path}/batch_1.xlsx")
    return pd.concat([df0, df1], ignore_index=True)

df_eyelink = load_and_merge_batches(eyelink_path)
df_phone = load_and_merge_batches(phone_path)

# 匹配subj_id
common_ids = set(df_eyelink['subj_id']) & set(df_phone['subj_id'])
df_eyelink = df_eyelink[df_eyelink['subj_id'].isin(common_ids)].sort_values('subj_id').reset_index(drop=True)
df_phone = df_phone[df_phone['subj_id'].isin(common_ids)].sort_values('subj_id').reset_index(drop=True)

# 去掉ID列，保留公共特征
features_eyelink = df_eyelink.drop(columns=['subj_id'])
features_phone = df_phone.drop(columns=['subj_id'])
common_features = [f for f in (set(features_eyelink.columns) & set(features_phone.columns))]
features_eyelink = features_eyelink[common_features]
features_phone = features_phone[common_features]

# ===== Nature配色 =====
nature_palette = {
    'fv': '#018A67',  # 深蓝
    'sp': '#1f77b4',  # 柔红
    'fs': "#d62728",  # 柔绿
}
title_map = {'fv': "Free viewing", 'sp': "Smooth pursuit", 'fs': "Fixation stability"}

# ===== 绘图函数 =====
def plot_nature_correlation(prefix, color, ax):
    selected_cols = [col for col in common_features if col.startswith(prefix)]
    if not selected_cols:
        print(f"No features found with prefix '{prefix}'")
        return

    corr_vals = features_eyelink[selected_cols].corrwith(features_phone[selected_cols])

    sns.histplot(
        corr_vals,
        bins=15,
        kde=True,
        color=color,
        linewidth=1.5,
        stat='density',
        alpha=0.85,
        ax=ax,
        fill=False,  # 关键参数：不填充直方图
        edgecolor=color,  # 边框颜色
        # histtype='step',
    )

    mean_corr = corr_vals.mean()
    median_corr = corr_vals.median()
    min_val, max_val = corr_vals.min(), corr_vals.max()

    ax.axvline(mean_corr, color='gray', linestyle='--', linewidth=1, label=f'Mean = {mean_corr:.2f}')
    ax.axvline(median_corr, color='black', linestyle='-.', linewidth=1, label=f'Median = {median_corr:.2f}')
    # ax.plot(min_val, 0, 'v', color='gray')
    # ax.plot(max_val, 0, '^', color='gray')
    # ax.text(min_val, 0.01, f'{min_val:.2f}', ha='center', va='bottom', fontsize=6, color='gray')
    # ax.text(max_val, 0.01, f'{max_val:.2f}', ha='center', va='bottom', fontsize=6, color='gray')

    ax.set_title(title_map.get(prefix, prefix), pad=8)
    ax.set_xlabel("Pearson Correlation Coefficient")
    if prefix == 'fv':
        ax.set_ylabel("Density")
    else:
        ax.set_ylabel("")
    ax.set_xlim(-0.2, 0.6)
    ax.set_ylim(0, 9)
    ax.legend(loc='upper left', frameon=False)

    # 添加统计框
    stats_text = (f'N = {len(corr_vals)}\n'
                  f'Std = {corr_vals.std():.2f}')
    ax.text(0.76, 0.95, stats_text,
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=6,
            bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=3.0))


# ===== 创建三图拼接 =====
fig, axs = plt.subplots(1, 3,)
fig.subplots_adjust(wspace=0.2)
for i, prefix in enumerate(['fv', 'sp', 'fs']):
    plot_nature_correlation(prefix, nature_palette[prefix], axs[i])

plt.tight_layout()
plt.savefig("fig_2/fig2_e.tiff", dpi=600, format='tiff')
# plt.show()

# 计算所有任务的整体 Pearson correlation 的平均值
all_corr_vals = []

for prefix in ['fv', 'sp', 'fs']:
    selected_cols = [col for col in common_features if col.startswith(prefix)]
    corr_vals = features_eyelink[selected_cols].corrwith(features_phone[selected_cols])
    all_corr_vals.extend(corr_vals.values)

overall_mean_corr = pd.Series(all_corr_vals).mean()
overall_median_std = pd.Series(all_corr_vals).std()

print(f"\n=== Overall Mean Pearson Correlation (All Tasks) ===\nMean = {overall_mean_corr:.4f}\nSTD = {overall_median_std:.4f}")
import os

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

# ==== 修改为你的真实路径 ====
from depression_symptom_detection import dep_dir
from schizophrenia_detection import sz_dir


def get_sig_label_and_p_text(p):
    """生成带有显著性标记和斜体p值的文本"""
    if p < 0.001:
        return '***', r'$\mathit{p < 0.001}$'
    elif p < 0.01:
        return '**', r'$\mathit{p = ' + f'{p:.3f}' + r'}$'
    elif p < 0.05:
        return '*', r'$\mathit{p = ' + f'{p:.3f}' + r'}$'
    else:
        return 'ns', r'$\mathit{n.s.}$'


def winsorize(data, limits=[0.00, 0.95]):
    """Trim extreme values at specified percentiles"""
    lower, upper = np.percentile(data, limits)
    return np.clip(data, lower, upper)


def cohen_d_and_ci(g1, g2):
    diff = np.mean(g1) - np.mean(g2)
    pooled_std = np.sqrt(((np.std(g1, ddof=1) ** 2 + np.std(g2, ddof=1) ** 2) / 2))
    d = diff / pooled_std

    rng = np.random.default_rng(42)
    boot_d = []
    for _ in range(5000):
        sample1 = rng.choice(g1, size=len(g1), replace=True)
        sample2 = rng.choice(g2, size=len(g2), replace=True)
        pooled = np.sqrt(((np.std(sample1, ddof=1) ** 2 + np.std(sample2, ddof=1) ** 2) / 2))
        boot_d.append((np.mean(sample1) - np.mean(sample2)) / pooled)

    ci_low, ci_high = np.percentile(boot_d, [2.5, 97.5])
    return d, (ci_low, ci_high)


def compute_stats(df, metric):
    g1 = df[df['label'] == 0][metric].dropna().values
    g2 = df[df['label'] == 1][metric].dropna().values

    n1, n2 = len(g1), len(g2)

    # 方差齐性检验
    # stat_levene, p_levene = levene(g1, g2)
    # equal_var = p_levene > 0.05  # 如果 p > 0.05，接受方差相等假设

    # t 检验
    t_stat, p_val = ttest_ind(g1, g2, equal_var=False)

    # # 自由度计算
    # if equal_var:
    #     df_value = n1 + n2 - 2
    # else:
    #     # Welch-Satterthwaite 近似公式
    #     s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    #     numerator = (s1 / n1 + s2 / n2) ** 2
    #     denominator = ((s1 / n1) ** 2) / (n1 - 1) + ((s2 / n2) ** 2) / (n2 - 1)
    #     df_value = numerator / denominator
    df_value = n1 + n2 - 2
    d, (ci_low, ci_high) = cohen_d_and_ci(g1, g2)

    return t_stat, p_val, df_value, d, (ci_low, ci_high), g1, g2


def box_plotting(ax, df, metric, title, group_labels, ylim=None):
    palette = {'0': '#1f77b4', '1': '#d62728'}  # 蓝红，适合Nature风格
    df = df.copy()

    sns.stripplot(
        data=df, x='label',  y=metric, ax=ax,
        palette=palette, size=1, jitter=0.2, alpha=0.3
    )

    sns.boxplot(
        data=df, x='label', y=metric, ax=ax, palette=palette,
        showfliers=False, width=0.60, linewidth=1, saturation=1,
        fill=False,
        whiskerprops=dict(linestyle='--'),  # 虚线
        # medianprops=dict(linewidth=2),
        # boxprops=dict(alpha=0.8)
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels([group_labels[0], group_labels[1]], fontsize=6)
    ax.set_ylabel(f'{metric.replace("mean_", "").capitalize()}', fontsize=6)
    ax.set_xlabel("", fontsize=7)
    if ylim is not None:
        ax.set_ylim(ylim)
    # ax.set_title(title, fontsize=14, fontweight='bold')
    sns.despine(ax=ax, trim=True)


if __name__ == "__main__":
    # ============================================================
    # plot configuration
    # ============================================================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    plt.rcParams['font.size'] = 7
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['figure.dpi'] = 600

    data_quality_dir = os.path.join(os.path.dirname(__file__), 'data_quality')
    fig_dir = os.path.join(os.path.dirname(__file__), 'fig_2')
    os.makedirs(fig_dir, exist_ok=True)

    # ==== 加载抑郁症数据 ====
    dep_df = pd.read_csv(f"{data_quality_dir}/dep_phone_data_quality.csv")
    dep_df['id'] = dep_df['id'].astype(str).str.lower()
    dep_meta = pd.read_excel(f'{dep_dir}/ground_truth/depression_symptom_label.xlsx')
    dep_meta['subj'] = dep_meta['subj'].astype(str).str.lower()
    dep_group_mapping = dep_meta.set_index('subj')['overlap_depression'].to_dict()
    dep_df.loc[:, 'label'] = dep_df['id'].map(dep_group_mapping).astype(int)

    # ==== 加载精神分裂症数据 ====
    sz_meta_file = f'{sz_dir}/meta_data/meta_data_release.xlsx'
    sz_meta = pd.concat([pd.read_excel(sz_meta_file, sheet_name=f"batch_{i}") for i in [0, 1]])
    sz_meta['id'] = sz_meta['id'].astype(str).str.lower()

    sz_phone_df = pd.read_csv(os.path.join(data_quality_dir, "sz_phone_data_quality.csv"))
    sz_phone_df['id'] = sz_phone_df['id'].astype(str).str.lower()
    sz_group_mapping = sz_meta.set_index('id')['sz'].to_dict()
    sz_phone_df.loc[:, 'label'] = sz_phone_df['id'].map(sz_group_mapping).astype(int)

    # ==== 创建图像 ====
    # fig, axs = plt.subplots(1, 4, figsize=(7.09, 2.4))
    # plt.subplots_adjust(hspace=0.4, wspace=0.3)

    titles = [
        'Depression',
        'Depression',
        'Schizophrenia',
        'Schizophrenia'
    ]
    study_titles = [
        'Depression study (Study 2)',
        'Schizophrenia study (Study 1)',
    ]

    datasets = [
        (dep_df, 'mean_accuracy', "Depression", {0: 'Non-Symptomatic', 1: 'Symptomatic'}),
        (dep_df, 'mean_precision', "Depression", {0: 'Non-Symptomatic', 1: 'Symptomatic'}),
        (sz_phone_df, 'mean_accuracy', "Schizophrenia", {0: 'Healthy controls', 1: 'Schizophrenia'}),
        (sz_phone_df, 'mean_precision', "Schizophrenia", {0: 'Healthy controls', 1: 'Schizophrenia'})
    ]

    # ===== 分为两个 1x2 图像 =====
    for fig_idx, offset in enumerate([0, 2]):
        fig, axs = plt.subplots(1, 2, figsize=(3.4, 2.4))  # 每张图大小适配2个图

        for j in range(2):
            i = offset + j
            ax = axs[j]
            df, metric, label, groups = datasets[i]
            t_stat, p_val, df_welch, d, ci, g1, g2 = compute_stats(df, metric)

            if metric == 'mean_accuracy':
                df = df[df[metric] <= 10]
                ylim = (0, 7)
                box_plotting(ax, df, metric, titles[i], group_labels=groups, ylim=ylim)
                y_max = 7 * 0.88
                h = y_max * 0.05
            else:
                ylim = (0, 0.5)
                box_plotting(ax, df, metric, titles[i], group_labels=groups, ylim=ylim)
                y_max = 0.5 * 0.88
                h = y_max * 0.05

            bar_y = y_max + h
            text_y = y_max + h * 1.1
            ax.plot([0, 0, 1, 1], [y_max, bar_y, bar_y, y_max], lw=1.4, c='k')
            sig_label, p_text = get_sig_label_and_p_text(p_val)
            ax.text(0.5, text_y, f"{p_text}", ha='center', va='bottom', fontsize=7)

            # 控制台输出
            print("\n" + "=" * 50)
            print(f"{titles[i]}:")
            print(f"Group 0: Median = {np.median(g1):.2f}, Mean={np.mean(g1):.2f}, Std={np.std(g1):.2f}, N={len(g1)}")
            print(f"Group 1: Median = {np.median(g2):.2f}, Mean={np.mean(g2):.2f}, Std={np.std(g2):.2f}, N={len(g2)}")
            print(f"t = {t_stat:.3f}, df = {df_welch:.1f}, p = {p_val:.4f}, d = {d:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]")
        fig.text(0.5, -0.05, study_titles[fig_idx], ha='center', va='bottom', fontsize=7)

        # 保存每一张图
        plt.tight_layout()
        fig_id = 'a' if fig_idx == 1 else 'b'
        fig_path = os.path.join(fig_dir, f'fig2_{fig_id}.tiff')
        plt.savefig(fig_path, dpi=600, bbox_inches='tight')
        # plt.show()

    # # 保存
    # save_path = os.path.join(fig_dir, 'nature_style_boxplot_2x2.tiff')
    # plt.tight_layout()
    # plt.savefig(save_path, dpi=600, bbox_inches='tight')
    # plt.show()

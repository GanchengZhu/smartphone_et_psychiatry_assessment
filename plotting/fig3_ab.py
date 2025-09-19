# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from depression_symptom_detection import dep_dir
from schizophrenia_detection import sz_dir

# 保存路径
save_dir = 'fig_3'
os.makedirs(save_dir, exist_ok=True)

# 设置统一样式
plt.rcParams.update({
    'font.family': 'Arial',
    'mathtext.fontset': 'stix',
    'font.size': 7,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'figure.dpi': 600
})

# 分类器及映射
classifiers = ['bagging', 'catboost', 'dtree', 'knn', 'lr', 'nb', 'rf', 'svm']
classifier_name_map = {
    'bagging': 'BAG', 'catboost': 'CB', 'dtree': 'DT', 'knn': 'KNN',
    'lr': 'LR', 'nb': 'NB', 'rf': 'RF', 'svm': 'SVM'
}
metric_label_map = {
    'accuracy': 'Accuracy',
    'precision': 'Precision',
    'recall': 'Recall'
}
metrics_plotting = list(metric_label_map.keys())
# NATURE_COLORS = ['#257D8B', '#EAA558', '#68BED9', '#ED8D5A']

NATURE_COLORS = [
    "#018A67",  # 柔绿
    '#d62728',  # 柔红
    '#1f77b4',  # 深蓝
]
plt.set_cmap("coolwarm")

# =========================
# 数据预处理函数
# =========================
def load_results(base_path, devices):
    results = {f'{clf}_{device}': [] for device in devices for clf in classifiers}
    for device in devices:
        seed_path = os.path.join(base_path, device, '42')
        for clf in classifiers:
            clf_path = os.path.join(seed_path, clf, 'k_fold.csv')
            if os.path.exists(clf_path):
                df = pd.read_csv(clf_path)
                df_sorted = df.sort_values(by=['val_recall', 'val_f1', 'val_auc', 'val_mcc', 'val_accuracy'],
                                           ascending=False)
                best_row = df_sorted.iloc[0]
                results[f'{clf}_{device}'].append(best_row.to_dict())
            else:
                raise Exception(f"Not found {clf_path}")
    return results


def prepare_plot_data(results_dict, device, label):
    data = []
    for clf in classifiers:
        key = f'{clf}_{device}'
        if results_dict[key]:
            df_row = pd.DataFrame(results_dict[key]).iloc[0]
            for metric in metrics_plotting:
                data.append({
                    'Classifier': classifier_name_map[clf],
                    'Metric': metric_label_map[metric],
                    'Score': df_row.get(metric),
                    'Device': label
                })
    df = pd.DataFrame(data)
    df['Metric'] = pd.Categorical(df['Metric'], categories=['Accuracy', 'Precision', 'Recall'], ordered=True)
    return df


# =========================
# 图像排版参数
# =========================
fig_width = 7.09
fig_height = 1.96
margin_in = 0.4
total_margin = 3 * margin_in
w1 = (fig_width - total_margin) / 3
h = 0.80
bottom = 0.10
m = margin_in / fig_width
w1_frac = w1 / fig_width

# 子图坐标
a1_pos = [0 + 2 * m, bottom, w1_frac, h]
a2_pos = [w1_frac + 3 * m, bottom, w1_frac, h]
b_pos = [2 * w1_frac + 4.5 * m, bottom, w1_frac, h]

# =========================
# 准备数据
# =========================
sz_base = f'{sz_dir}/results/test'
dep_base = f'{dep_dir}/results/test'

sz_results = load_results(sz_base, ['phone', 'eyelink'])
dep_results = load_results(dep_base, ['overlap_depression'])

df_a1 = prepare_plot_data(sz_results, 'phone', 'Smartphone, SZ')
df_a2 = prepare_plot_data(sz_results, 'eyelink', 'EyeLink, SZ')
df_b = prepare_plot_data(dep_results, 'overlap_depression', 'Depression')


# =========================
# 绘图函数
# =========================
def plot_bar(df, ax, show_ylabel=False, label_text=''):
    sns.despine(ax=ax)
    sns.barplot(
        data=df,
        x='Classifier',
        y='Score',
        hue='Metric',
        # palette=NATURE_COLORS,
        ax=ax,
        linewidth=1.0,
        saturation=1,
        width=0.5,  # 缩小 bar 的宽度，间距会变大
        alpha=1,
        dodge=True,
        fill=False,
    )

    for j in range(1, len(classifiers)):
        ax.axvline(x=j - 0.5, color='gray', linestyle=':', alpha=0.3, linewidth=0.6)

    ax.set_ylim(0, 1)
    ax.set_yticks([i / 10 for i in range(11)])
    ax.set_yticklabels([f'{i / 10:.1f}' for i in range(11)])
    ax.set_xlabel('')
    ax.set_ylabel('Score' if show_ylabel else '')
    ax.tick_params(labelsize=6)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
    ax.set_title('')
    ax.text(0.5, -0.2, label_text, fontsize=7, transform=ax.transAxes,
            ha='center', va='top')
    ax.get_legend().remove()


# =========================
# 创建组合图像
# =========================
fig = plt.figure(figsize=(fig_width, fig_height))

ax_a1 = fig.add_axes(a1_pos)
ax_a2 = fig.add_axes(a2_pos)
ax_b = fig.add_axes(b_pos)

plot_bar(df_a1, ax_a1, show_ylabel=True, label_text='Schizophrenia | Smartphone')
plot_bar(df_a2, ax_a2, show_ylabel=False, label_text='Schizophrenia | EyeLink')
plot_bar(df_b, ax_b, show_ylabel=True, label_text='Depression | Smartphone')

# 添加统一图例
handles, labels = ax_a1.get_legend_handles_labels()
fig.legend(handles, labels, title='', loc='upper center', ncol=3, fontsize=6, frameon=False,
           bbox_to_anchor=(0.425, 1.02))
fig.legend(handles, labels, title='', loc='upper center', ncol=3, fontsize=6, frameon=False,
           bbox_to_anchor=(0.945, 1.02))

fig.savefig(f"{save_dir}/fig3_ab.tiff", dpi=600, bbox_inches='tight')
plt.close(fig)

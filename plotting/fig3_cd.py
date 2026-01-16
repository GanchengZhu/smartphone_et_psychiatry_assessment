import json
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

from depression_symptom_detection import data_reader as dep_reader, dep_dir
from schizophrenia_detection import data_reader as sz_reader, sz_dir

# 设置参数
plt.rcParams.update({
    'font.family': 'Arial',
    'mathtext.fontset': 'stix',
    'font.size': 7,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'figure.dpi': 600
})

# 保存路径
save_dir = 'fig_3'
os.makedirs(save_dir, exist_ok=True)

# 分类器列表与映射
classifiers = ['emprotonet', 'catboost', 'dtree', 'knn', 'lr', 'nb', 'rf', 'svm', 'bagging', ]
classifier_name_map = {
    'bagging': 'BAG', 'catboost': 'CB', 'dtree': 'DT', 'knn': 'KNN',
    'lr': 'LR', 'nb': 'NB', 'rf': 'RF', 'svm': 'SVM', 'emprotonet': 'EMPN'
}
metrics_priority = ['val_recall', 'val_f1', 'val_auc', 'val_mcc', 'val_accuracy']

# palette = nature_palette = [
#     "#1f77b4",  # 蓝色
#     "#d62728",  # 红色
#     "#2ca02c",  # 绿色
#     "#ff7f0e",  # 橙色
#     "#9467bd",  # 紫色
#     "#8c564b",  # 棕色
#     "#17becf",  # 青色
#     "#bcbd22",  # 黄绿色
#     "#e377c2",  # 粉色
#     "#7f7f7f",  # 灰色
#     "#aec7e8",  # 浅蓝
#     "#ff9896"   # 浅红
# ]

plt.set_cmap("coolwarm")


# -----------------------
# 工具函数
# -----------------------
def load_hyper_parameters(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)


def bootstrap_roc_ci(y_true, y_score, n_bootstraps=1000, seed=42):
    rng = np.random.RandomState(seed)
    tprs = []
    base_fpr = np.linspace(0, 1, 100)
    aucs = []

    for _ in range(n_bootstraps):
        indices = rng.choice(len(y_score), len(y_score), replace=True)
        if len(np.unique(y_true[indices])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[indices], y_score[indices])
        aucs.append(roc_auc_score(y_true[indices], y_score[indices]))
        tpr_interp = np.interp(base_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    tprs = np.array(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    mean_auc = np.mean(aucs)
    lower = np.percentile(aucs, 2.5)
    upper = np.percentile(aucs, 97.5)
    return base_fpr, mean_tpr, std_tpr, mean_auc, lower, upper


def draw_roc_with_ci_from_npz(path):
    data = np.load(path)
    predictions = data['predictions']
    labels = data['labels']
    probabilities = data['probabilities']
    # confidences = data['confidences']
    data.close()
    if probabilities.ndim == 2:
        probabilities = probabilities[:, 1]
    return bootstrap_roc_ci(np.array(labels), np.array(probabilities))


def plot_roc_subplot(ax, device_name, param_base_dir, reader_module,
                     device_label, show_y_lable=True):
    import seaborn as sns
    sns.despine(ax=ax)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Chance')
    for n, clf in enumerate(classifiers):
        npz_path = os.path.join(param_base_dir, 'results', 'test', device_name, '42', clf, 'test_results.npz')
        result = draw_roc_with_ci_from_npz(npz_path)

        if result is None:
            continue
        base_fpr, mean_tpr, std_tpr, mean_auc, lower, upper = result
        label = f"{classifier_name_map[clf]} (AUC = {mean_auc * 100:.2f}% [{lower * 100:.2f}%–{upper * 100:.2f}%])"
        # print(label)
        ax.plot(base_fpr, mean_tpr, lw=1.0, label=label, alpha=0.8,
                # color=palette[n]
                )
        print(f"{device_name}: {label}")
        # ax.fill_between(base_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, alpha=0.15)

    ax.set_xlabel('False Positive Rate')
    if show_y_lable:
        ax.set_ylabel('True Positive Rate')
    ax.set_title('')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.text(0.5, -0.22, device_label, fontsize=7, transform=ax.transAxes,
            ha='center', va='top')
    ax.legend(fontsize=4, loc='lower right', framealpha=0, edgecolor='none')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)


fig = plt.figure(figsize=(7.09, 2.4))
# 位置计算
margin_in = 0.35
total_margin = 3 * margin_in
w1 = (7.09 - total_margin) / 3
h = 0.80
bottom = 0.10
m = margin_in / 7.09
w1_frac = w1 / 7.09
positions = [
    [0 + 2 * m, bottom, w1_frac, h],
    [w1_frac + 3.2 * m, bottom, w1_frac, h],
    [2 * w1_frac + 5 * m, bottom, w1_frac, h]
]

# 子图绘制
axes = [fig.add_axes(pos) for pos in positions]
plot_roc_subplot(axes[0], 'phone', sz_dir, sz_reader,
                 'Schizophrenia | Smartphone')
plot_roc_subplot(axes[1], 'eyelink', sz_dir, sz_reader,
                 'Schizophrenia | EyeLink', show_y_lable=False)
plot_roc_subplot(axes[2], 'overlap_depression', dep_dir, dep_reader,
                 'Depression | Smartphone')

fig.savefig(f'{save_dir}/fig3_cd.tiff', dpi=600, bbox_inches='tight')
plt.close(fig)

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from depression_symptom_detection import data_reader as dep_reader, dep_dir
from schizophrenia_detection_test_retest import data_reader as sz_reader, sz_dir

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
classifiers = ['bagging', 'catboost', 'dtree', 'knn', 'lr', 'nb', 'rf', 'svm']
classifier_name_map = {
    'bagging': 'BAG', 'catboost': 'CB', 'dtree': 'DT', 'knn': 'KNN',
    'lr': 'LR', 'nb': 'NB', 'rf': 'RF', 'svm': 'SVM'
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


def get_classifier_map(random_state):
    return {
        'svm': SVC(kernel='rbf', C=1.0, random_state=random_state, probability=True),
        'lr': LogisticRegression(random_state=random_state),
        'knn': KNeighborsClassifier(n_jobs=-1),
        'nb': GaussianNB(),
        'dtree': DecisionTreeClassifier(random_state=random_state),
        'rf': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'lgbm': LGBMClassifier(),
        'bagging': BaggingClassifier(n_estimators=100, random_state=random_state),
        'xgboost': XGBClassifier(),
        'catboost': CatBoostClassifier(random_seed=random_state, verbose=0),
    }


def draw_roc_with_ci(train_X, train_y, test_X, test_y, param_path, fold_id, classifier_name, random_state=42):
    sd = StandardScaler()
    train_X = sd.fit_transform(train_X)
    test_X = sd.transform(test_X)

    clf_map = get_classifier_map(random_state)
    classifier = clf_map[classifier_name]
    pipeline = Pipeline([('classifier', classifier)])

    if not os.path.exists(param_path):
        print(f"[警告] 未找到参数文件: {param_path}")
        return None

    params = load_hyper_parameters(param_path)
    pipeline.set_params(**params)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    for fold_idx, (train_idx, _) in enumerate(skf.split(train_X, train_y)):
        if fold_idx + 1 != fold_id:
            continue
        X_fold = train_X[train_idx]
        y_fold = train_y[train_idx]
        pipeline.fit(X_fold, y_fold)

        if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
            y_score = pipeline.predict_proba(test_X)[:, 1]
        else:
            y_score = pipeline.decision_function(test_X)

        return bootstrap_roc_ci(np.array(test_y), np.array(y_score))


def plot_roc_subplot(ax, train_X, train_y, test_X, test_y, device_name, param_base_dir, fold_id_dict, reader_module,
                     device_label, show_y_lable=True):
    import seaborn as sns
    sns.despine(ax=ax)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, label='Chance')
    for n, clf in tqdm.tqdm(enumerate(classifiers)):
        param_path = os.path.join(param_base_dir, 'results', 'train', device_name, '42', clf, 'best_params.json')
        key = f'{clf}_{device_name}'
        fold_id = fold_id_dict.get(key, 1)
        # print(param_path)
        # print(fold_id)
        result = draw_roc_with_ci(train_X, train_y, test_X, test_y, param_path, fold_id, clf)
        if result is None:
            continue
        base_fpr, mean_tpr, std_tpr, mean_auc, lower, upper = result
        label = f"{classifier_name_map[clf]} (AUC = {mean_auc:.2f} [{lower:.2f}–{upper:.2f}])"
        # print(label)
        ax.plot(base_fpr, mean_tpr, lw=1.0, label=label, alpha=0.8,
                # color=palette[n]
                )
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


# -----------------------
# 数据准备
# -----------------------
def prepare_fold_ids(base_dir, devices):
    fold_dict = {f'{clf}_{device}': 0 for device in devices for clf in classifiers}
    for device in devices:
        device_path = os.path.join(base_dir, device, '42')
        for clf in classifiers:
            csv_path = os.path.join(device_path, clf, 'k_fold.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                best_row = df.sort_values(by=metrics_priority, ascending=False).iloc[0]
                fold_dict[f'{clf}_{device}'] = best_row['fold']
    return fold_dict


# -----------------------
# 主图绘制
# -----------------------
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

# SZ数据
sz_fold_ids = prepare_fold_ids(f'{sz_dir}/results/test', ['phone', 'eyelink'])
X_phone, X_phone_test, y_phone, y_phone_test, *_ = sz_reader.split_data(data_source='phone', random_seed=42)
X_eye, X_eye_test, y_eye, y_eye_test, *_ = sz_reader.split_data(data_source='eyelink', random_seed=42)

# Depression数据
dep_fold_ids = prepare_fold_ids(f'{dep_dir}/results/test', ['overlap_depression'])
X_dep, X_dep_test, y_dep, y_dep_test, *_ = dep_reader.split_data(data_source='overlap_depression', random_seed=42,
                                                                 test_size=0.2)

# 子图绘制
axes = [fig.add_axes(pos) for pos in positions]
plot_roc_subplot(axes[0], X_phone, y_phone, X_phone_test, y_phone_test, 'phone', sz_dir, sz_fold_ids, sz_reader,
                 'Schizophrenia | Smartphone')
plot_roc_subplot(axes[1], X_eye, y_eye, X_eye_test, y_eye_test, 'eyelink', sz_dir, sz_fold_ids, sz_reader,
                 'Schizophrenia | EyeLink', show_y_lable=False)
plot_roc_subplot(axes[2], X_dep, y_dep, X_dep_test, y_dep_test, 'overlap_depression', dep_dir, dep_fold_ids, dep_reader,
                 'Depression | Smartphone')

fig.savefig(f'{save_dir}/fig3_cd.tiff', dpi=600, bbox_inches='tight')
plt.close(fig)

# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from schizophrenia_detection import sz_dir, data_reader

# ------------------ 配置 ------------------
classifiers = ['bagging', 'catboost', 'dtree', 'knn', 'lr', 'nb', 'rf', 'svm']
classifier_name_map = {
    'bagging': 'BAG', 'catboost': 'CB', 'dtree': 'DT', 'knn': 'KNN',
    'lr': 'LR', 'nb': 'NB', 'rf': 'RF', 'svm': 'SVM'
}
metrics_priority = ['val_recall', 'val_f1', 'val_auc', 'val_mcc', 'val_accuracy']

# Nature style parameters
plt.rcParams.update({
    'font.family': 'Arial',
    'mathtext.fontset': 'stix',
    'font.size': 6,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'legend.frameon': False,
    'legend.fontsize': 5
})

save_dir = os.path.join(os.path.dirname(__file__), 'ex_fig_3')
os.makedirs(save_dir, exist_ok=True)


# ------------------ 加载超参数 ------------------
def load_hyper_parameters(best_param_json_path):
    with open(best_param_json_path, 'r') as f:
        return json.load(f)


# ------------------ 训练+预测函数 ------------------
def ml(train_X, train_y, rm_X, rm_y, device, random_seed, classifier, fold_id_dict):
    sd = StandardScaler()
    train_X = sd.fit_transform(train_X)
    rm_X = sd.transform(rm_X)

    s_k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_seed)

    classifier_map = {
        'svm': SVC(kernel='rbf', C=1.0, random_state=random_seed, probability=True),
        'lr': LogisticRegression(random_state=random_seed),
        'knn': KNeighborsClassifier(n_jobs=-1),
        'nb': GaussianNB(),
        'dtree': DecisionTreeClassifier(random_state=random_seed),
        'rf': RandomForestClassifier(n_estimators=100, random_state=random_seed),
        'lgbm': LGBMClassifier(),
        'bagging': BaggingClassifier(n_estimators=100, random_state=random_seed),
        'catboost': CatBoostClassifier(random_seed=random_seed, verbose=False),
        'xgboost': XGBClassifier()
    }

    clf_name = classifier
    classifier = classifier_map.get(classifier, XGBClassifier())
    pipeline = Pipeline([('classifier', classifier)])

    best_param_json_path = f"{sz_dir}/results/train/{device}/{str(random_seed)}/{clf_name}/best_params.json"
    if not os.path.exists(best_param_json_path):
        print(f"[警告] 未找到参数文件: {best_param_json_path}")
        return None

    param_dict = load_hyper_parameters(best_param_json_path)
    pipeline.set_params(**param_dict)

    for fold_idx, (train_idx, val_idx) in enumerate(s_k_fold.split(train_X, train_y)):
        if fold_idx + 1 != fold_id_dict[f'{clf_name}_{device}']:
            continue
        X_fold_train, y_fold_train = train_X[train_idx], train_y[train_idx]
        pipeline.fit(X_fold_train, y_fold_train)

        y_pre = pipeline.predict(rm_X)
        if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
            y_score = pipeline.predict_proba(rm_X)[:, 1]
        else:
            y_score = pipeline.decision_function(rm_X)

        return rm_y, y_pre, y_score


# ------------------ 读取 fold 信息 ------------------
base_dir = f'{sz_dir}/results/test'
random_seed = 42
fold_id_dict = {f'{clf}_{device}': 0 for device in ['phone', 'eyelink'] for clf in classifiers}

for device in ['phone', 'eyelink']:
    device_path = os.path.join(base_dir, device, str(random_seed))
    for clf in classifiers:
        clf_path = os.path.join(device_path, clf, 'k_fold.csv')
        if os.path.exists(clf_path):
            df = pd.read_csv(clf_path)
            df_sorted = df.sort_values(by=metrics_priority, ascending=False)
            best_row = df_sorted.iloc[0]
            fold_id_dict[f'{clf}_{device}'] = best_row['fold']

# ------------------ 计算 Precision 和 Recall ------------------
metrics_dict = {
    'phone': {'test': {'precision': [], 'recall': []}, 'retest': {'precision': [], 'recall': []}},
    'eyelink': {'test': {'precision': [], 'recall': []}, 'retest': {'precision': [], 'recall': []}}
}

for device in ['phone', 'eyelink']:
    X_train, X_test, y_train, y_test, X_retest, y_retest, _ = data_reader.split_data(
        data_source=device, random_seed=random_seed
    )

    for classifier in tqdm.tqdm(classifiers):
        # Test
        result_test = ml(
            train_X=X_train, train_y=y_train,
            rm_X=X_test, rm_y=y_test,
            device=device, random_seed=random_seed,
            classifier=classifier, fold_id_dict=fold_id_dict
        )
        if result_test is not None:
            y_true, y_pre, _ = result_test
            metrics_dict[device]['test']['recall'].append(recall_score(y_true, y_pre) * 100)
            metrics_dict[device]['test']['precision'].append(precision_score(y_true, y_pre) * 100)
        else:
            # 添加占位值
            metrics_dict[device]['test']['recall'].append(0)
            metrics_dict[device]['test']['precision'].append(0)

        # Retest
        result_retest = ml(
            train_X=X_train, train_y=y_train,
            rm_X=X_retest, rm_y=y_retest,
            device=device, random_seed=random_seed,
            classifier=classifier, fold_id_dict=fold_id_dict
        )
        if result_retest is not None:
            y_true, y_pre, _ = result_retest
            metrics_dict[device]['retest']['recall'].append(recall_score(y_true, y_pre) * 100)
            metrics_dict[device]['retest']['precision'].append(precision_score(y_true, y_pre) * 100)
        else:
            # 添加占位值
            metrics_dict[device]['retest']['recall'].append(0)
            metrics_dict[device]['retest']['precision'].append(0)

# ------------------ 绘制 2x2 Nature 风格子图 ------------------
# 为不同设备定义不同的颜色方案
phone_colors = {
    'test': '#1f77b4',  # 深蓝色
    'retest': '#aec7e8'  # 浅蓝色
}

eyelink_colors = {
    'test': '#d62728',  # 深红色
    'retest': '#ff9896'  # 浅红色
}

# 创建图形
fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.2))
bar_width = 0.35
x = np.arange(len(classifiers))

sub_labels = ['a', 'b', 'c', 'd']
plot_info = [
    ('Precision', 'phone', 0, 0),
    ('Precision', 'eyelink', 0, 1),
    ('Recall', 'phone', 1, 0),
    ('Recall', 'eyelink', 1, 1)
]

for idx, (metric, device, row_idx, col_idx) in enumerate(plot_info):
    ax = axs[row_idx, col_idx]

    # 选择颜色方案
    colors = phone_colors if device == 'phone' else eyelink_colors

    # 获取数据
    test_data = metrics_dict[device]['test'][metric.lower()]
    retest_data = metrics_dict[device]['retest'][metric.lower()]

    # 绘制柱状图
    bars_test = ax.bar(x - bar_width / 2, test_data, bar_width,
                       label='Test', color=colors['test'], alpha=0.75)
    bars_retest = ax.bar(x + bar_width / 2, retest_data, bar_width,
                         label='Retest', color=colors['retest'], alpha=0.75)

    # 数值标注
    # 数值标注（带自动错开逻辑）
    for i, (test_val, retest_val) in enumerate(zip(test_data, retest_data)):
        offset_test = 1.0
        offset_retest = 1.0
        # 如果两个柱子差距很小，避免重叠 → 自动错开
        if abs(test_val - retest_val) < 3:
            offset_test = 5.0
            offset_retest = 1.0

        ax.text(i - bar_width / 2, test_val + offset_test, f'{test_val:.2f}',
                ha='center', va='bottom', fontsize=5)
        ax.text(i + bar_width / 2, retest_val + offset_retest, f'{retest_val:.2f}',
                ha='center', va='bottom', fontsize=5)

    # 设置轴标签和标题
    ax.set_xticks(x)
    ax.set_xticklabels([classifier_name_map[c] for c in classifiers], rotation=45)
    ax.set_ylabel(f'{metric} (%)')
    ax.set_ylim(30, 100)

    # 添加子图标签 (a, b, c, d)
    ax.text(-0.17, 1.10, sub_labels[idx], transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top')

    # 添加设备名称
    device_name = 'Smartphone' if device == 'phone' else 'EyeLink'
    ax.text(0.5, -0.18, device_name, transform=ax.transAxes,
            fontsize=6, ha='center', va='top', fontweight='bold')

    # 设置网格
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)

    # 移除上右边框
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # 在每个子图内添加图例
    ax.legend(loc='upper right', fontsize=5, frameon=False)

plt.tight_layout()
plt.savefig(f'{save_dir}/precision_recall_2x2_nature.tiff', format='tiff', bbox_inches='tight')
# plt.savefig(f'{save_dir}/precision_recall_2x2_nature.pdf', format='pdf', bbox_inches='tight')
plt.close()
print(metrics_dict)

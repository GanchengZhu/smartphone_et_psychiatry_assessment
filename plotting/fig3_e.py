import json
import os
import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from matplotlib.colors import LinearSegmentedColormap

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from schizophrenia_detection import data_reader, sz_dir
from schizophrenia_detection.variable_map import sz_feature_name_map

classifier_name_map = {
    'rf': 'RF',
    'catboost': 'CB',
    'bagging': "BAG",
    'lr': "LR",
    'svm': "SVM"
}

device_label_map = {
    'phone': 'iPhone',
    'eyelink': 'EyeLink'
}

# 你的颜色列表
colors = [
    "#1f77b4",
    "#6997c7",
    "#9eb8da",
    "#cfdbec",
    "#ffffff",
    "#ffcec5",
    "#f89c8d",
    "#ea6959",
    "#d62728"
]

custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)

random_state = 42
save_dir = os.path.join(os.path.dirname(__file__), 'fig_3')
os.makedirs(save_dir, exist_ok=True)


def load_hyper_parameters(best_param_json_path):
    if not os.path.exists(best_param_json_path):
        warnings.warn(f"未找到参数文件: {best_param_json_path}")
        return {}

    try:
        with open(best_param_json_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        warnings.warn(f"参数文件解析失败: {best_param_json_path}")
        return {}


def train_model(
        X_train: pd.DataFrame,
        train_y: pd.Series,
        device: str,
        classifier: str,
):
    sd = StandardScaler()
    X_train_scaled = sd.fit_transform(X_train)

    classifier_map = {
        'svm': SVC(kernel='rbf', C=1.0, random_state=random_state, probability=True),
        'lr': LogisticRegression(random_state=random_state, n_jobs=-1),
        'knn': KNeighborsClassifier(n_jobs=-1),
        'nb': GaussianNB(),
        'dtree': DecisionTreeClassifier(random_state=random_state),
        'rf': RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        'lgbm': LGBMClassifier(n_jobs=-1, random_state=random_state),
        'bagging': BaggingClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        'xgboost': XGBClassifier(),
        'catboost': CatBoostClassifier(eval_metric="F1", task_type='CPU', random_seed=random_state, verbose=False),
    }

    clf = classifier_map[classifier]
    pipeline = Pipeline([('classifier', clf)])
    best_param_json_path = f"{sz_dir}/results/train/{device}/42/{classifier}/best_params.json"
    param_dict = load_hyper_parameters(best_param_json_path)
    pipeline.set_params(**param_dict)
    pipeline.fit(X_train_scaled, train_y)

    return pipeline, sd


def evaluate_model(
        pipeline: Pipeline,
        standard_scaler: StandardScaler,
        test_X: pd.DataFrame,
        test_y: pd.Series
) -> dict:
    test_X_scaled = standard_scaler.transform(test_X)

    y_pred = pipeline.predict(test_X_scaled)
    y_prob = pipeline.predict_proba(test_X_scaled)[:, 1]

    acc = accuracy_score(test_y, y_pred)
    f1 = f1_score(test_y, y_pred)
    prec = precision_score(test_y, y_pred)
    recall = recall_score(test_y, y_pred)
    auc = roc_auc_score(test_y, y_prob)

    return {
        "Accuracy": acc,
        "F1 Score": f1,
        "Precision": prec,
        "Recall": recall,
        "AUC": auc
    }


def shap_summary_plot(
        pipeline,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        feature_names: List[str],
        save_path: str
):
    clf = pipeline.named_steps['classifier']

    # 自动选择合适的预测函数
    predict_fn = clf.predict_proba if hasattr(clf, "predict_proba") else clf.decision_function
    explainer = shap.Explainer(predict_fn, X_train, feature_names=feature_names)
    shap_values = explainer(X_test)

    # 二分类模型处理（取正类）
    if len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
        shap_values = shap_values[:, :, 1]

    feature_names = [
        a + ": " + str(b) for a, b in zip(feature_names, np.abs(shap_values.values).mean(0).round(3))
    ]
    # Matplotlib样式更新
    plt.rcParams.update({
        'font.family': 'Arial',
        'mathtext.fontset': 'stix',
        'font.size': 6,
        'axes.linewidth': 1,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'figure.dpi': 600
    })
    # 创建图像
    fig = plt.figure(figsize=(3.3, 2.8))

    # SHAP点图
    shap.summary_plot(
        shap_values,
        X_test,
        # show_values_in_legend=True,
        feature_names=feature_names,
        plot_type="dot",
        show=False,
        max_display=15,
        alpha=0.5,
        color=plt.get_cmap("coolwarm")  # 或直接使用字符串 "coolwarm"
        # color=custom_cmap
    )

    # ax = plt.gca()
    # for label in ax.get_xticklabels() + ax.get_yticklabels():
    #     label.set_fontweight('bold')
    plt.xlim((-0.125, 0.125))
    # 紧凑布局并保存
    plt.tight_layout(pad=0.5)  # 进一步减小内边距
    plt.savefig(save_path, dpi=600, bbox_inches='tight', transparent=False)
    plt.close(fig)  # 明确关闭图形

# -------------------
# 🚀 主程序
# -------------------
def main():
    random_seed = 42
    targets = [('phone', 'svm'), ('eyelink', 'svm')]

    for device, classifier in targets:
        X_train, X_test, y_train, y_test, _, _, feature_names = data_reader.split_data(
            data_source=device, random_seed=random_seed)

        new_feature_names = [sz_feature_name_map[feature] for feature in feature_names]

        pipeline, sd = train_model(X_train, y_train, device, classifier)
        eval_metrics = evaluate_model(pipeline, sd, X_test, y_test)
        X_train_scaled = sd.transform(X_train)
        X_test_scaled = sd.transform(X_test)
        # 📊 SHAP 特征重要性
        shap_save_path = f'{save_dir}/fig3_e_{device}.tiff'
        shap_summary_plot(
            pipeline, X_train_scaled, X_test_scaled, new_feature_names,
            shap_save_path,
        )

        print(f"\n📊 Test Evaluation - {device_label_map[device]} ({classifier_name_map[classifier]})")
        for metric, value in eval_metrics.items():
            print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()

# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import argparse
import json
import logging
import os
import sys

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, accuracy_score, balanced_accuracy_score, roc_auc_score, \
    average_precision_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import data_reader


def load_hyper_parameters(best_param_json_path):
    logger.info(f"Load best hyperparameters from {best_param_json_path}")
    with open(best_param_json_path, 'r') as f:
        best_param_dict = json.load(f)
        return best_param_dict


def test(train_X, train_y, test_X, test_y, args):
    logger.info(f"Starting training process for {args.classifier}")
    logger.info(f"Data source: {args.data_source}")

    # 创建输出目录
    sd = StandardScaler()
    train_X = sd.fit_transform(train_X)
    test_X = sd.transform(test_X)

    s_k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Classifier initialization
    classifier_map = {
        'svm': SVC(kernel='rbf', C=1.0, random_state=42, probability=True),
        'lr': LogisticRegression(random_state=42),
        'knn': KNeighborsClassifier(n_jobs=-1),
        'nb': GaussianNB(),
        'dtree': DecisionTreeClassifier(random_state=42),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'lgbm': LGBMClassifier(),
        'bagging': BaggingClassifier(n_estimators=100, random_state=42),
        'xgboost': XGBClassifier(),
        'catboost': CatBoostClassifier(random_seed=42, verbose=0),
    }

    classifier = classifier_map.get(args.classifier, XGBClassifier())
    logger.info(f"Initialized classifier: {classifier.__class__.__name__}")

    pipeline = Pipeline([
        ('classifier', classifier)
    ])

    logger.debug("Created pipeline with feature selector and classifier")
    best_param_json_path = f"results/train/{args.data_source}/{str(args.random_seed)}/{args.classifier}/best_params.json"
    param_dict = load_hyper_parameters(best_param_json_path)
    print(param_dict)
    pipeline.set_params(**param_dict)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(s_k_fold.split(train_X, train_y)):
        # 划分数据
        X_train, y_train = train_X[train_idx], train_y[train_idx]
        # X_val, y_val = train_X[val_idx], train_y[val_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(test_X)
        y_proba = pipeline.predict_proba(test_X)[:, 1] if hasattr(classifier, 'predict_proba') else [0] * len(test_X)

        X_val, y_val = train_X[val_idx], train_y[val_idx]
        y_pred_val = pipeline.predict(X_val)
        y_proba_val = pipeline.predict_proba(X_val)[:, 1] if hasattr(classifier, 'predict_proba') else [0] * len(X_val)

        # 计算指标
        metrics = {
            'fold': fold_idx + 1,
            'accuracy': accuracy_score(test_y, y_pred),
            'balanced_accuracy': balanced_accuracy_score(test_y, y_pred),
            'mcc': matthews_corrcoef(test_y, y_pred),
            'auc': roc_auc_score(test_y, y_proba) if y_proba[0] != 0 else 0.0,
            'aupr': average_precision_score(test_y, y_proba) if y_proba[0] != 0 else 0.0,
            'f1': f1_score(test_y, y_pred),
            'precision': precision_score(test_y, y_pred),
            'recall': recall_score(test_y, y_pred),
            'val_accuracy': accuracy_score(y_val, y_pred_val),
            'val_balanced_accuracy': balanced_accuracy_score(y_val, y_pred_val),
            'val_mcc': matthews_corrcoef(y_val, y_pred_val),
            'val_auc': roc_auc_score(y_val, y_proba_val) if y_proba_val[0] != 0 else 0.0,
            'val_aupr': average_precision_score(y_val, y_proba_val) if y_proba_val[0] != 0 else 0.0,
            'val_f1': f1_score(y_val, y_pred_val),
            'val_precision': precision_score(y_val, y_pred_val),
            'val_recall': recall_score(y_val, y_pred_val)
        }
        fold_results.append(metrics)
        logger.info(f"Fold {fold_idx + 1} completed")

    # 保存交叉验证结果
    cv_output_dir = os.path.join(f"results/test/{args.data_source}/{args.random_seed}", args.classifier)
    os.makedirs(cv_output_dir, exist_ok=True)
    pd.DataFrame(fold_results).to_csv(os.path.join(cv_output_dir, 'k_fold.csv'), index=False)
    logger.info(f"Saved CV results to {cv_output_dir}/k_fold.csv")


if __name__ == '__main__':
    # Parse arguments first
    parser = argparse.ArgumentParser(description='Train a sklearn classifier with feature selection.')
    parser.add_argument('--classifier', type=str, required=True,
                        choices=['svm', 'lr', 'knn', 'nb', 'dtree', 'rf', 'bagging', 'catboost'], )
    parser.add_argument('--data_source', choices=['phone', 'eyelink'], type=str, required=True)
    parser.add_argument('--random_seed', type=int, default=42, required=False)
    # parser.add_argument('--gpu', type=str, default='0', help='Set GPU device ID')
    args = parser.parse_args()

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # 设置日志级别

    os.makedirs(f'./log/{args.data_source}/{args.random_seed}/{args.classifier}', exist_ok=True)
    # 创建文件处理器（保存到文件）
    file_handler = logging.FileHandler(f"./log/{args.data_source}/{args.random_seed}/{args.classifier}_test.log",
                                       mode="a")  # 保存到 app.log 文件
    file_handler.setLevel(logging.INFO)  # 文件日志级别

    # 创建控制台处理器（可选：同时输出到终端）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # 控制台日志级别

    # 定义日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)  # 如果不需要控制台输出，可注释此行

    X_train, X_test, y_train, y_test, X_rm, y_rm, feature_names = data_reader.split_data(data_source=args.data_source,
                                                                                         random_seed=args.random_seed)
    test(train_X=X_train, train_y=y_train, test_X=X_test, test_y=y_test, args=args)
    logger.info("Testing completed successfully")

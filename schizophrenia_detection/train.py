# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import argparse
import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    print("Using Intel-optimized scikit-learn")
except ImportError:
    print("scikit-learn-intelex not found, using standard scikit-learn")

from catboost import CatBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import data_reader


def train(X, y, args):
    logger.info(f"Starting training process for {args.classifier}")
    logger.info(f"Data source: {args.data_source}")

    output_path = f"results/train/{args.data_source}/{str(args.random_seed)}/{args.classifier}"
    os.makedirs(output_path, exist_ok=True)
    logger.debug(f"Created output directory: {output_path}")

    X = StandardScaler().fit_transform(X)
    s_k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.random_seed)

    # Classifier initialization
    classifier_map = {
        'svm': SVC(kernel='rbf', C=1.0, random_state=args.random_seed, probability=True),
        'lr': LogisticRegression(random_state=args.random_seed, n_jobs=-1),
        'knn': KNeighborsClassifier(n_jobs=-1),
        'nb': GaussianNB(),
        'dtree': DecisionTreeClassifier(random_state=args.random_seed),
        'rf': RandomForestClassifier(n_estimators=100, random_state=args.random_seed, n_jobs=-1),
        'bagging': BaggingClassifier(n_estimators=100, random_state=args.random_seed, n_jobs=-1),
        'catboost': CatBoostClassifier(eval_metric="F1", task_type='CPU', random_seed=args.random_seed, verbose=False),
    }

    classifier = classifier_map.get(args.classifier, None)
    logger.info(f"Initialized classifier: {classifier.__class__.__name__}")

    pipeline = Pipeline([
        ('classifier', classifier)
    ])
    logger.debug("Created pipeline with feature selector and classifier")

    # 参数网格
    param_grid = {}

    # 分类器参数
    if args.classifier == 'svm':
        param_grid.update({
            'classifier__C': np.logspace(-3, 3, 200),
            'classifier__kernel': ['rbf'],
            'classifier__gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7)),
        })

    elif args.classifier == 'lr':
        param_grid.update({
            'classifier__C': np.logspace(-4, 3, 1000),
            'classifier__penalty': ['l1', 'l2', 'elasticnet'],
            'classifier__solver': ['saga'],
            # 'classifier__penalty': ['l2'],
        })

    elif args.classifier == 'knn':
        param_grid.update({
            'classifier__n_neighbors': [i for i in range(1, 11)],
            'classifier__p': list(range(1, 6)),
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski'],
        })

    elif args.classifier == 'nb':
        param_grid.update({
            'classifier__var_smoothing': [1e-9, 1e-8, 1e-7]
        })

    elif args.classifier == 'dtree':
        param_grid.update({
            'classifier__max_depth': [3, 5, 7, 9, 13, 17, None],
            'classifier__min_samples_split': [2, 5, 10, 15, 21],
            'classifier__min_samples_leaf': [1, 2, 4, 8, 16],
            'classifier__criterion': ['gini', 'entropy'],
        })

    elif args.classifier == 'rf':
        param_grid.update({
            'classifier__n_estimators': [100, 200, 300, 400, 500],
            'classifier__max_depth': [5, 10, 15, 20, 25, 30, None],
            'classifier__min_samples_split': [2, 5, 10, 20],
            'classifier__min_samples_leaf': [1, 2, 4, 8],
        })

    elif args.classifier == 'bagging':
        param_grid.update({
            'classifier__n_estimators': [50, 100, 200, 500],
            'classifier__max_samples': [0.5, 0.7, 1.0],
            'classifier__max_features': [0.5, 0.7, 1.0],
        })

    elif args.classifier == 'catboost':
        param_grid.update({
            'classifier__learning_rate': [0.01, 0.05, 0.1],  # 控制梯度下降步长
            'classifier__iterations': [100, 200, 500],  # 最大树数量
            'classifier__depth': [2, 4, 6],  # 树深度(替代num_leaves)
            # 'classifier__min_data_in_leaf': [1, 5, 10]  # 叶节点最小数据量
        })

    else:
        pass

    # logger.info("Starting grid search with parameters:\n%s", json.dumps(param_grid, indent=2))
    # Grid Search
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=s_k_fold,
        scoring='roc_auc',
        refit=False,
        n_jobs=-1,
        verbose=0
    )
    logger.info("Fitting GridSearchCV...")
    y = y.astype(int)
    grid_search.fit(X, y)
    logger.info(f"Best parameters found: {grid_search.best_params_}")

    # Save results
    best_params_path = f"{output_path}/best_params.json"
    with open(best_params_path, 'w') as f:
        json.dump(grid_search.best_params_, f)
    logger.info(f"Saved best parameters to {best_params_path}")


if __name__ == '__main__':
    # Parse arguments first
    parser = argparse.ArgumentParser(description='Train a sklearn classifier')
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
    file_handler = logging.FileHandler(f"./log/{args.data_source}/{args.random_seed}/{args.classifier}/train.log",
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
    train(X=X_train, y=y_train, args=args)
    logger.info("Training completed successfully")

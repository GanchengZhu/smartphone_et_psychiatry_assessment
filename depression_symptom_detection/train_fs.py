# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Define metrics
scoring = {
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'mcc': make_scorer(matthews_corrcoef),
    'auc': 'roc_auc',
    'aupr': 'average_precision',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall',
}


def select_top_n_by_mi(X, y, n_keep, random_state):
    """Select top-n features based on mutual information"""
    mi = mutual_info_classif(X, y, discrete_features='auto', random_state=random_state)
    top_indices = np.argsort(mi)[::-1][:n_keep]
    return top_indices


def train(X, y, args):
    logger.info(f"Starting training process for {args.classifier}")

    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / 'results' / 'train' / args.data_source / str(args.random_seed) / args.classifier
    output_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created output directory: {output_path}")

    X = StandardScaler().fit_transform(X)
    top_indices = select_top_n_by_mi(X, y, args.n_feature_keep, args.random_seed)
    X = X[:, top_indices]

    s_k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.random_seed)

    classifier_map = {
        'svm': SVC(kernel='rbf', C=1.0, random_state=args.random_seed, probability=True),
        'lr': LogisticRegression(random_state=args.random_seed, n_jobs=-1),
        'knn': KNeighborsClassifier(n_jobs=-1),
        'nb': GaussianNB(),
        'dtree': DecisionTreeClassifier(random_state=args.random_seed),
        'rf': RandomForestClassifier(n_estimators=100, random_state=args.random_seed, n_jobs=-1),
        'lgbm': LGBMClassifier(n_jobs=-1, random_state=args.random_seed),
        'bagging': BaggingClassifier(n_estimators=100, random_state=args.random_seed, n_jobs=-1),
        'xgboost': XGBClassifier(),
        'catboost': CatBoostClassifier(eval_metric="F1", task_type='CPU', random_seed=args.random_seed, verbose=False),
    }

    classifier = classifier_map.get(args.classifier, XGBClassifier())
    logger.info(f"Initialized classifier: {classifier.__class__.__name__}")

    pipeline = Pipeline([
        ('classifier', classifier)
    ])
    logger.debug("Created pipeline with classifier")

    param_grid = {}

    if args.classifier == 'svm':
        param_grid.update({
            'classifier__C': np.logspace(-3, 3, 200),
            'classifier__kernel': ['linear', 'rbf', 'sigmoid'],
            'classifier__gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7)),
        })

    elif args.classifier == 'lr':
        param_grid.update({
            'classifier__C': np.logspace(-4, 3, 1000),
            'classifier__penalty': ['l1', 'l2', 'elasticnet'],
            'classifier__solver': ['saga'],
        })

    elif args.classifier == 'knn':
        param_grid.update({
            'classifier__n_neighbors': list(range(1, 11)),
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

    elif args.classifier == 'lgbm':
        param_grid.update({
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__n_estimators': [100, 200, 300],
            'classifier__num_leaves': [15, 31, 63],
            'classifier__max_bin': [200, 255],
            'classifier__colsample_bytree': [0.6, 0.8, 1.0],
            'classifier__subsample': [0.6, 0.8, 1.0],
            'classifier__reg_alpha': [0, 0.05, 0.1, 0.5, 1],
            'classifier__reg_lambda': [0, 0.05, 0.1, 0.5, 1],
            'classifier__max_depth': [5, 7, -1],
            'classifier__min_child_samples': [10, 20]
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
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__iterations': [100, 200, 500],
            'classifier__depth': [2, 4, 6],
        })

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
    grid_search.fit(X, y)
    logger.info(f"Best parameters found: {grid_search.best_params_}")

    # Save best parameters
    best_params_path = output_path / 'best_params.json'
    with open(best_params_path, 'w') as f:
        json.dump(grid_search.best_params_, f)

    logger.info(f"Saved best parameters to {best_params_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a sklearn classifier')
    parser.add_argument('--classifier', type=str, required=True,
                        choices=['svm', 'lr', 'knn', 'nb', 'dtree', 'rf', 'bagging', 'catboost'])
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--data_source', type=str, default="overlap_depression")
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--n_feature_keep', type=int, default=150)
    args = parser.parse_args()

    # Initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / 'log' / args.data_source / str(args.random_seed) / args.classifier
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "train.log", mode="a")
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    import data_reader

    X_train, X_test, y_train, y_test, _ = data_reader.split_data(
        data_source=args.data_source,
        random_seed=args.random_seed,
        test_size=args.test_size
    )

    logger.info(f"X_train shape is {X_train.shape}")
    logger.info(f"y_train shape is {y_train.shape}")
    logger.info(f"X_test shape is {X_test.shape}")
    logger.info(f"y_test shape is {y_test.shape}")

    train(X=X_train, y=y_train, args=args)
    logger.info("Training completed successfully")

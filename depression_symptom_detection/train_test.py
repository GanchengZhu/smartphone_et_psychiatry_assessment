# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from sklearnex import patch_sklearn

    patch_sklearn()
    print("Using Intel-optimized scikit-learn")
except ImportError:
    print("scikit-learn-intelex not found, using standard scikit-learn")

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_selection import (
    SelectKBest, RFE, SelectFromModel, f_classif
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import data_reader


def setup_logger(log_dir: Path, mode: str = "train"):
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{mode}_logger")
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_dir / f"{mode}.log", mode="a")
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def train(X, y, args, logger):
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / 'results' / 'train' / args.data_source / str(args.random_seed)
    if args.demographic_separation:
        output_path /= args.classifier + '_demographic_separation'
    else:
        output_path /= args.classifier

    output_path.mkdir(parents=True, exist_ok=True)

    X = StandardScaler().fit_transform(X)
    s_k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=args.random_seed)

    classifier_map = {
        'svm': SVC(kernel='rbf', C=1.0, random_state=args.random_seed, probability=True),
        'lr': LogisticRegression(random_state=args.random_seed, n_jobs=-1, max_iter=1000),
        'knn': KNeighborsClassifier(n_jobs=-1),
        'nb': GaussianNB(),
        'dtree': DecisionTreeClassifier(random_state=args.random_seed),
        'rf': RandomForestClassifier(n_estimators=100, random_state=args.random_seed, n_jobs=-1),
        'lgbm': LGBMClassifier(n_jobs=-1, random_state=args.random_seed),
        'bagging': BaggingClassifier(n_estimators=100, random_state=args.random_seed, n_jobs=-1),
        'catboost': CatBoostClassifier(eval_metric="F1", task_type='CPU', random_seed=args.random_seed, verbose=False),
    }

    classifier = classifier_map.get(args.classifier)
    logger.info(f"Initialized classifier: {classifier.__class__.__name__}")

    pipeline_steps = []
    logger.info("Adding feature selection to pipeline")
    if args.feature_selection_method == 'selectkbest':
        pipeline_steps.append(('feature_selection', SelectKBest(score_func=f_classif)))
    elif args.feature_selection_method == 'selectfrommodel':
        estimator = RandomForestClassifier(n_estimators=50, random_state=args.random_seed)
        pipeline_steps.append(('feature_selection', SelectFromModel(estimator=estimator, threshold='median')))
    elif args.feature_selection_method == 'rfe':
        estimator = LogisticRegression(random_state=args.random_seed, max_iter=1000)
        pipeline_steps.append(('feature_selection', RFE(estimator=estimator, step=0.1)))

    pipeline_steps.append(('classifier', classifier))
    pipeline = Pipeline(pipeline_steps)

    param_grid = {}

    if args.feature_selection_method == 'selectkbest':
        # param_grid.update({'feature_selection__k':  list(range(20, 125, 5))})
        param_grid.update({'feature_selection__k': [120]})
    elif args.feature_selection_method == 'selectfrommodel':
        param_grid.update({
            'feature_selection__estimator__n_estimators': [50, 100],
            'feature_selection__threshold': ['mean', 'median', 0.1, 0.5]
        })

    if args.classifier == 'svm':
        param_grid.update({
            'classifier__C': np.logspace(-3, 3, 20),
            'classifier__kernel': ['rbf'],
            'classifier__gamma': ['scale', 'auto'] + list(np.logspace(-3, 3, 7)),
        })
    elif args.classifier == 'lr':
        param_grid.update({
            'classifier__C': np.logspace(-4, 3, 50),
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
        param_grid.update({'classifier__var_smoothing': [1e-9, 1e-8, 1e-7]})
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
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [5, 10, 15, 20, None],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
        })
    elif args.classifier == 'bagging':
        param_grid.update({
            'classifier__n_estimators': [50, 100, 200],
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
        refit=True,  # 关键修复：refit=True 以便直接使用 best_estimator_
        n_jobs=-1,
        verbose=0
    )

    logger.info("Fitting GridSearchCV...")
    grid_search.fit(X, y)
    logger.info(f"Best parameters found: {grid_search.best_params_}")
    logger.info(f"Best CV AUC: {grid_search.best_score_:.4f}")

    # 保存最佳模型（可选）
    # joblib.dump(grid_search.best_estimator_, output_path / 'best_model.pkl')

    best_params_path = output_path / 'best_params.json'
    with open(best_params_path, 'w') as f:
        json.dump(grid_search.best_params_, f)
    logger.info(f"Saved best parameters to {best_params_path}")

    # 如果使用了 pipeline 中的特征选择，保存所选特征
    if 'feature_selection' in grid_search.best_estimator_.named_steps:
        selector = grid_search.best_estimator_.named_steps['feature_selection']
        if hasattr(selector, 'get_support'):
            selected_features = selector.get_support()
            selected_indices = np.where(selected_features)[0]
            feature_selection_info = {
                'n_selected_features': len(selected_indices),
                'selected_indices': selected_indices.tolist(),
                'feature_selection_method': args.feature_selection_method
            }
            with open(output_path / 'feature_selection_info.json', 'w') as f:
                json.dump(feature_selection_info, f)
            # logger.info(f"Saved feature selection info to {feature_selection_path}")


def apply_feature_selection(X_train, X_test, feature_selection_info, logger):
    if not feature_selection_info:
        return X_train, X_test

    selected_indices = [int(i) for i in feature_selection_info.get('selected_indices', [])]
    method = feature_selection_info.get('feature_selection_method', '')

    if not selected_indices:
        return X_train, X_test

    if max(selected_indices) >= X_train.shape[1]:
        logger.warning("Selected indices out of range; filtering...")
        selected_indices = [i for i in selected_indices if i < X_train.shape[1]]

    X_train_selected = X_train[:, selected_indices]
    X_test_selected = X_test[:, selected_indices]
    logger.info(f"Applied {method}: selected {len(selected_indices)} features")
    return X_train_selected, X_test_selected


def test(train_X, train_y, test_X, test_y, feature_names, args, logger):
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / 'results' / 'test' / args.data_source / str(args.random_seed)
    if args.demographic_separation:
        output_path /= args.classifier + '_demographic_separation'
    else:
        output_path /= args.classifier
    output_path.mkdir(parents=True, exist_ok=True)

    sd = StandardScaler()
    train_X = sd.fit_transform(train_X)
    test_X = sd.transform(test_X)

    classifier_map = {
        'svm': SVC(kernel='rbf', C=1.0, random_state=args.random_seed, probability=True),
        'lr': LogisticRegression(random_state=args.random_seed, n_jobs=-1, max_iter=1000),
        'knn': KNeighborsClassifier(n_jobs=-1),
        'nb': GaussianNB(),
        'dtree': DecisionTreeClassifier(random_state=args.random_seed),
        'rf': RandomForestClassifier(n_estimators=100, random_state=args.random_seed, n_jobs=-1),
        'lgbm': LGBMClassifier(n_jobs=-1, random_state=args.random_seed),
        'bagging': BaggingClassifier(n_estimators=100, random_state=args.random_seed, n_jobs=-1),
        'catboost': CatBoostClassifier(eval_metric="F1", task_type='CPU', random_seed=args.random_seed, verbose=False),
    }
    classifier = classifier_map.get(args.classifier)
    logger.info(f"Initialized classifier: {classifier.__class__.__name__}")

    pipeline_steps = []
    best_params_path = script_dir / 'results' / 'train' / args.data_source / str(
        args.random_seed) / args.classifier / 'best_params.json'
    with open(best_params_path, 'r') as f:
        param_dict = json.load(f)

    if args.feature_selection_method == 'selectkbest':
        k = param_dict.get('feature_selection__k', 'all')
        pipeline_steps.append(('feature_selection', SelectKBest(k=k, score_func=f_classif)))
    elif args.feature_selection_method == 'selectfrommodel':
        n_est = param_dict.get('feature_selection__estimator__n_estimators', 100)
        thresh = param_dict.get('feature_selection__threshold', 'median')
        est = RandomForestClassifier(n_estimators=n_est, random_state=args.random_seed)
        pipeline_steps.append(('feature_selection', SelectFromModel(estimator=est, threshold=thresh)))
    elif args.feature_selection_method == 'rfe':
        est = LogisticRegression(random_state=args.random_seed, max_iter=1000)
        n_feat = param_dict.get('feature_selection__n_features_to_select', None)
        if n_feat is not None:
            n_feat = int(n_feat)
        pipeline_steps.append(('feature_selection', RFE(estimator=est, n_features_to_select=n_feat, step=0.1)))

    pipeline_steps.append(('classifier', classifier))
    pipeline = Pipeline(pipeline_steps)

    para_path = script_dir / 'results' / 'train' / args.data_source / str(
        args.random_seed) / args.classifier / 'best_params.json'
    with open(para_path, 'r') as f:
        param_dict = json.load(f)
    pipeline.set_params(**param_dict)

    pipeline.fit(train_X, train_y)
    feature_selector = pipeline.named_steps['feature_selection']
    selected_indices = feature_selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]

    if args.feature_selection_method == 'selectkbest':
        if hasattr(feature_selector, 'scores_'):
            # 获取特征得分
            feature_scores = feature_selector.scores_

    selection_info = {
        'selected_features': selected_features,
        'selected_indices': selected_indices.tolist(),
        'num_selected_features': len(selected_features)
    }

    with open(output_path / 'selected_features.json', 'w') as f:
        json.dump(selection_info, f, indent=2)

    y_pred = pipeline.predict(test_X)
    y_proba = pipeline.predict_proba(test_X)[:, 1] if hasattr(pipeline, 'predict_proba') else np.zeros(len(test_X))

    def safe_score(func, y_true, y_score):
        try:
            return func(y_true, y_score)
        except Exception as e:
            logger.warning(f"Metric failed: {e}")
            return 0.0

    metrics = {
        'accuracy': accuracy_score(test_y, y_pred),
        'balanced_accuracy': balanced_accuracy_score(test_y, y_pred),
        'mcc': matthews_corrcoef(test_y, y_pred),
        'auc': safe_score(roc_auc_score, test_y, y_proba),
        'aupr': safe_score(average_precision_score, test_y, y_proba),
        'f1': f1_score(test_y, y_pred),
        'precision': precision_score(test_y, y_pred),
        'recall': recall_score(test_y, y_pred),
    }

    results_df = pd.DataFrame([metrics])
    results_df.to_csv(output_path / 'test_results.csv', index=False)

    np.savez_compressed(
        output_path / 'test_results.npz',
        predictions=np.array(y_pred),
        labels=np.array(test_y),
        probabilities=np.array(y_proba),
    )

    with open(output_path / 'test_results.json', 'w') as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Test AUC: {metrics['auc']:.4f}")

    # 特征重要性（如果有的话）
    if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
        importances = pipeline.named_steps['classifier'].feature_importances_
        with open(output_path / 'feature_importance_test.json', 'w') as f:
            json.dump({'importance': importances.tolist()}, f, indent=4)

    return metrics['auc']


def main():
    parser = argparse.ArgumentParser(description='Train and test a sklearn classifier with feature selection.')
    parser.add_argument('--classifier', type=str, required=True,
                        choices=['svm', 'lr', 'knn', 'nb', 'dtree', 'rf', 'bagging', 'catboost'])
    parser.add_argument('--test_size', type=float, default=0.25, help="Proportion of test set (0 < test_size < 1)")
    parser.add_argument('--data_source', type=str, default="overlap_depression")
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--feature_selection_method', type=str, default='selectkbest',
                        choices=['selectkbest', 'selectpercentile', 'rfe', 'rfecv', 'selectfrommodel', 'variance'])
    parser.add_argument('--n_features', type=str, default='auto')
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--demographic_separation', action='store_true')

    args = parser.parse_args()

    if not (0 < args.test_size < 1):
        raise ValueError("--test_size must be between 0 and 1 (exclusive).")

    if args.n_features != 'auto':
        if '.' in args.n_features:
            args.n_features = float(args.n_features)
        else:
            args.n_features = int(args.n_features)
    else:
        args.n_features = None

    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / 'log' / args.data_source / str(args.random_seed) / args.classifier

    train_logger = setup_logger(log_dir, "train")
    test_logger = setup_logger(log_dir, "test")

    X_train, X_test, y_train, y_test, feature_names = data_reader.split_data(
        data_source=args.data_source,
        random_seed=args.random_seed,
        test_size=args.test_size,
        demographic_separation=args.demographic_separation,
    )

    train_logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    train_logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    if not args.test_only:
        # --- TRAIN ---
        train(X_train, y_train, args, train_logger)
        train_logger.info("Training completed successfully")

    # --- TEST ---
    test_auc = test(X_train, y_train, X_test, y_test, feature_names, args, test_logger)
    test_logger.info("Testing completed successfully")


if __name__ == '__main__':
    main()

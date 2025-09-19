# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    make_scorer, matthews_corrcoef, accuracy_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import data_reader


def load_hyper_parameters(para_path):
    logger.info(f"Load best hyperparameters from {para_path}")
    with open(para_path, 'r') as f:
        best_param_dict = json.load(f)
        return best_param_dict


def test(train_X, train_y, test_X, test_y, args):
    logger.info(f"Starting testing process for {args.classifier}")

    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / 'results' / 'test' / args.data_source / str(args.random_seed) / args.classifier
    output_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Created output directory: {output_path}")

    sd = StandardScaler()
    train_X = sd.fit_transform(train_X)
    test_X = sd.transform(test_X)

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
        'catboost': CatBoostClassifier(eval_metric="F1", task_type='CPU', random_seed=args.random_seed, verbose=False),
    }

    classifier = classifier_map.get(args.classifier, None)
    logger.info(f"Initialized classifier: {classifier.__class__.__name__}")

    pipeline = Pipeline([
        ('classifier', classifier)
    ])

    logger.debug("Created pipeline with classifier")

    para_path = script_dir / 'results' / 'train' / args.data_source / str(
        args.random_seed) / args.classifier / 'best_params.json'
    param_dict = load_hyper_parameters(para_path)
    print(param_dict)
    pipeline.set_params(**param_dict)

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(s_k_fold.split(train_X, train_y)):
        train_X_fold, train_y_fold = train_X[train_idx], train_y[train_idx]

        pipeline.fit(train_X_fold, train_y_fold)
        y_pred = pipeline.predict(test_X)
        y_proba = pipeline.predict_proba(test_X)[:, 1] if hasattr(classifier, 'predict_proba') else [0] * len(test_X)

        val_X_fold, val_y_fold = train_X[val_idx], train_y[val_idx]
        y_pred_val = pipeline.predict(val_X_fold)
        y_proba_val = pipeline.predict_proba(val_X_fold)[:, 1] if hasattr(classifier, 'predict_proba') else [0] * len(val_X_fold)

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
            'val_accuracy': accuracy_score(val_y_fold, y_pred_val),
            'val_balanced_accuracy': balanced_accuracy_score(val_y_fold, y_pred_val),
            'val_mcc': matthews_corrcoef(val_y_fold, y_pred_val),
            'val_auc': roc_auc_score(val_y_fold, y_proba_val) if y_proba_val[0] != 0 else 0.0,
            'val_aupr': average_precision_score(val_y_fold, y_proba_val) if y_proba_val[0] != 0 else 0.0,
            'val_f1': f1_score(val_y_fold, y_pred_val),
            'val_precision': precision_score(val_y_fold, y_pred_val),
            'val_recall': recall_score(val_y_fold, y_pred_val)
        }
        fold_results.append(metrics)
        logger.info(f"Fold {fold_idx + 1} completed")

    pd.DataFrame(fold_results).to_csv(output_path / 'k_fold.csv', index=False)
    logger.info(f"Saved CV results to {output_path / 'k_fold.csv'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a sklearn classifier with best parameters.')
    parser.add_argument('--classifier', type=str, required=True,
                        choices=['svm', 'lr', 'knn', 'nb', 'dtree', 'rf', 'bagging', 'catboost'])
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--data_source', type=str, default="overlap_depression")
    parser.add_argument('--random_seed', type=int, default=42)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / 'log' / args.data_source / str(args.random_seed) / args.classifier
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "test.log", mode="a")
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

    X_train, X_test, y_train, y_test, _ = data_reader.split_data(
        data_source=args.data_source,
        random_seed=args.random_seed,
        test_size=args.test_size
    )

    test(train_X=X_train, train_y=y_train, test_X=X_test, test_y=y_test, args=args)
    logger.info("Testing completed successfully")

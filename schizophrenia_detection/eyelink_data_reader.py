# -*- coding: utf-8 -*-
"""
Author: GC Zhu
Email: zhugc2016@gmail.com
Description: Process eyelink-based features and metadata for seizure status prediction.
             Includes batch0, batch1, and repeat-measure data loading, cleaning, and train/test split.
"""

import os

import numpy as np
import pandas as pd


def load_and_clean_batch(meta_filepath: str,
                         sheet_name: str,
                         feature_filepath: str,
                         id_col: str = 'id',
                         subj_col: str = 'subj_id',
                         flag_col: str = 'eyelink_both',
                         target_col: str = 'sz',
                         random_state: int = 42, ):
    """
    Load metadata and features for a given batch, filter by eyelink_both flag,
    align features by subject ID, and return feature matrix X and target vector y.
    """
    # Read feature data
    feature_df = pd.read_excel(
        feature_filepath,
        dtype={id_col: 'int', subj_col: 'int'}
    )
    feature_df[subj_col] = feature_df[subj_col].astype(int)

    # Read metadata
    meta_df = pd.read_excel(
        meta_filepath,
        sheet_name=sheet_name,
        dtype={id_col: 'int', subj_col: 'int'}
    )

    if sheet_name == 'batch_0':
        # Step 1: Get all ids where phone_both & eyelink_both & is_repeat are True
        mask = (meta_df['phone_both']).astype(bool) & (meta_df['eyelink_both']).astype(bool) & (
                meta_df['is_repeat']).astype(bool)
        meta_retest_df = meta_df.loc[
            mask,
            [id_col, target_col]
        ]

        # Step 2: Randomly sample 30 ids from retest_id with a fixed random seed
        test_sz_df = meta_retest_df.sample(n=30, replace=False, random_state=random_state)

        # Step 3: From id range 095–188, get ids where phone_both & eyelink_both are True,
        # then randomly sample 30 with a fixed random seed
        meta_df_range = meta_df.loc[
            (meta_df[id_col].astype(int) >= 95) & (meta_df[id_col].astype(int) <= 188) &
            (meta_df['phone_both']) & (meta_df['eyelink_both']),
            [id_col, target_col]
        ]
        test_tc_df = meta_df_range.sample(n=30, replace=False, random_state=random_state)
        meta_test_df = pd.concat([test_sz_df, test_tc_df], axis=0)
        # the remaining ids are used for training
        mask = (
                meta_df[flag_col] &
                (~meta_df[id_col].isin(meta_test_df[id_col]))
        )
        meta_train_df = meta_df.loc[
            mask, [id_col, target_col]
        ]

        train_id = meta_train_df[id_col].astype(int)
        mask = feature_df[subj_col].isin(train_id)
        batch_0_X_train_df = feature_df.loc[mask].copy()
        group_mapping = meta_train_df.set_index(id_col)[target_col].to_dict()
        batch_0_y_train = batch_0_X_train_df[subj_col].map(group_mapping).values
        # Drop subject column and fill missing values
        batch_0_X_train_df.drop(columns=[subj_col], inplace=True)
        batch_0_X_train_df.fillna(0, inplace=True)
        # Ensure numeric type
        batch_0_X_train = batch_0_X_train_df.astype(np.float64).values

        test_sz_id = test_sz_df[id_col].astype(int)
        mask = feature_df[subj_col].isin(test_sz_id)
        batch_0_sz_X_test_df = feature_df.loc[mask].copy()
        group_mapping = test_sz_df.set_index(id_col)[target_col].to_dict()
        batch_0_sz_y_test = batch_0_sz_X_test_df[subj_col].map(group_mapping).values
        # Drop subject column and fill missing values
        batch_0_sz_X_test_df.drop(columns=[subj_col], inplace=True)
        batch_0_sz_X_test_df.fillna(0, inplace=True)
        # Ensure numeric type
        batch_0_sz_X_test = batch_0_sz_X_test_df.astype(np.float64).values

        test_tc_id = test_tc_df[id_col].astype(int)
        mask = feature_df[subj_col].isin(test_tc_id)
        batch_0_tc_X_test_df = feature_df.loc[mask].copy()
        group_mapping = test_tc_df.set_index(id_col)[target_col].to_dict()
        batch_0_tc_y_test = batch_0_tc_X_test_df[subj_col].map(group_mapping).values
        # Drop subject column and fill missing values
        batch_0_tc_X_test_df.drop(columns=[subj_col], inplace=True)
        batch_0_tc_X_test_df.fillna(0, inplace=True)
        # Ensure numeric type
        batch_0_tc_X_test = batch_0_tc_X_test_df.astype(np.float64).values

        feature_names = batch_0_X_train_df.columns.tolist()
        return (batch_0_X_train, batch_0_y_train,
                batch_0_sz_X_test, batch_0_sz_y_test,
                batch_0_tc_X_test, batch_0_tc_y_test,
                train_id, test_sz_id, test_tc_id, feature_names)

    elif sheet_name == 'batch_1':
        # Select records with complete phone data
        meta_df = meta_df[meta_df[flag_col] == 1].copy()
        # Clean ID columns (strip whitespace)
        meta_df[id_col] = meta_df[id_col].astype(int)

        # Keep only rows whose subject ID is in metadata
        mask = feature_df[subj_col].isin(meta_df[id_col])
        X_df = feature_df.loc[mask].copy()

        # group mapping
        sz_group_mapping = meta_df.set_index(id_col)[target_col].to_dict()
        y = X_df[subj_col].map(sz_group_mapping).values

        # Drop subject column and fill missing values
        X_df.drop(columns=[subj_col], inplace=True)
        X_df.fillna(0, inplace=True)

        # Ensure numeric type
        X = X_df.astype(np.float64).values
        feature_names = X_df.columns.tolist()
        return X, y, feature_names
    elif sheet_name == 'batch_1_sz_repeat_measure':
        meta_df_batch_0 = pd.read_excel(
            meta_filepath,
            sheet_name='batch_0',
            dtype={id_col: 'int', subj_col: 'int'}
        )
        # Step 1: Get all ids where phone_both & eyelink_both & is_repeat are True’
        mask = (
                meta_df_batch_0['phone_both'].astype(bool) &
                meta_df_batch_0['eyelink_both'].astype(bool) &
                meta_df_batch_0['is_repeat'].astype(bool)
        )
        meta_retest_df = meta_df_batch_0.loc[
            mask,
            [id_col, target_col]
        ]
        # Step 2: Randomly sample 30 ids from retest_id with a fixed random seed
        test_sz_df = meta_retest_df.sample(n=30, replace=False, random_state=random_state)

        # the remaining ids are used for training
        meta_train_df = meta_df.loc[
            meta_df[flag_col] &
            (~meta_df['corresponding_id'].isin(test_sz_df[id_col])),
            [id_col, target_col]
        ]

        meta_test_df = meta_df.loc[
            meta_df[flag_col] &
            (meta_df['corresponding_id'].isin(test_sz_df[id_col])),
            [id_col, target_col]
        ]

        train_id = meta_train_df[id_col].astype(int)
        mask = feature_df[subj_col].isin(train_id)
        batch_1_rm_X_train_df = feature_df.loc[mask].copy()
        group_mapping = meta_train_df.set_index(id_col)[target_col].to_dict()
        batch_1_rm_y_train = batch_1_rm_X_train_df[subj_col].map(group_mapping).values
        # Drop subject column and fill missing values
        batch_1_rm_X_train_df.drop(columns=[subj_col], inplace=True)
        batch_1_rm_X_train_df.fillna(0, inplace=True)
        # Ensure numeric type
        batch_1_rm_X_train = batch_1_rm_X_train_df.astype(np.float64).values

        test_id = meta_test_df[id_col].astype(int)
        mask = feature_df[subj_col].isin(test_id)
        batch_1_rm_X_test_df = feature_df.loc[mask].copy()
        group_mapping = meta_test_df.set_index(id_col)[target_col].to_dict()
        batch_1_rm_y_test = batch_1_rm_X_test_df[subj_col].map(group_mapping).values
        # Drop subject column and fill missing values
        batch_1_rm_X_test_df.drop(columns=[subj_col], inplace=True)
        batch_1_rm_X_test_df.fillna(0, inplace=True)
        # Ensure numeric type
        batch_1_rm_X_test = batch_1_rm_X_test_df.astype(np.float64).values

        feature_names = batch_1_rm_X_train_df.columns.tolist()
        return (batch_1_rm_X_train, batch_1_rm_y_train, batch_1_rm_X_test, batch_1_rm_y_test,
                train_id, test_id, feature_names)


def split_data(random_seed=42, retest_train_include=False):
    # File paths
    meta_path = os.path.join(os.path.dirname(__file__), 'meta_data', 'meta_data_release.xlsx')

    # Process batch 0
    (batch_0_X_train, batch_0_y_train,
     batch_0_sz_X_test, batch_0_sz_y_test,
     batch_0_tc_X_test, batch_0_tc_y_test,
     train_id, test_sz_id, test_tc_id, feature_names) = load_and_clean_batch(
        meta_filepath=meta_path,
        sheet_name='batch_0',
        feature_filepath=os.path.join(os.path.dirname(__file__), 'features', 'data_phone', 'batch_0.xlsx'),
        random_state=random_seed,
    )

    # Process batch 1
    (batch_1_X_train, batch_1_y_train, feature_names) = load_and_clean_batch(
        meta_filepath=meta_path,
        sheet_name='batch_1',
        feature_filepath=os.path.join(os.path.dirname(__file__), 'features', 'data_phone', 'batch_1.xlsx'),
        random_state=random_seed,
    )
    # print(f"Batch1: X shape = {batch1_X.shape}, y shape = {batch1_y.shape}")

    # Process repeat-measure data
    (batch_1_rm_X_train, batch_1_rm_y_train, batch_1_rm_X_test, batch_1_rm_y_test,
     batch_1_rm_train_id, batch_1_rm_test_id, feature_names) = load_and_clean_batch(
        meta_filepath=meta_path,
        sheet_name='batch_1_sz_repeat_measure',
        feature_filepath=os.path.join(os.path.dirname(__file__), 'features', 'data_phone',
                                      'batch_1_sz_repeat_measure.xlsx'),
        random_state=random_seed,
    )

    # Combine batch0 and batch1 data
    if retest_train_include:
        X_train = np.concatenate([batch_0_X_train, batch_1_X_train, batch_1_rm_X_train], axis=0)
        y_train = np.concatenate([batch_0_y_train, batch_1_y_train, batch_1_rm_y_train], axis=0)
    else:
        X_train = np.concatenate([batch_0_X_train, batch_1_X_train], axis=0)
        y_train = np.concatenate([batch_0_y_train, batch_1_y_train], axis=0)
    X_test = np.concatenate([batch_0_sz_X_test, batch_0_tc_X_test], axis=0)
    y_test = np.concatenate([batch_0_sz_y_test, batch_0_tc_y_test], axis=0)

    X_retest = np.concatenate([batch_1_rm_X_test, batch_0_tc_X_test], axis=0)
    y_retest = np.concatenate([batch_1_rm_y_test, batch_0_tc_y_test], axis=0)

    print("Typical controls:", np.sum(y_train == 0)
          + np.sum(y_test == 0), '\n', 'Schizophrenia:', np.sum(y_train == 1) + np.sum(y_test == 1))

    # Return or save train/test sets as needed
    return X_train, X_test, y_train, y_test, X_retest, y_retest, feature_names


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, X_rm, y_rm, _ = split_data()

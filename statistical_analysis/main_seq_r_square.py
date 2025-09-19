# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import os

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from schizophrenia_detection import sz_dir


def slope_func_duration(x, m, b):
    return m + b * x


# def sqrt_func_velocity(x, m):
#     return m * np.sqrt(x)
def power_func_velocity(x, m, a):
    return m * (x ** a)


def detect_outliers_gmm(data, n_components=2, threshold=0.05):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(data_scaled)
    densities = np.exp(gmm.score_samples(data_scaled))
    density_threshold = np.percentile(densities, threshold * 100)
    return densities >= density_threshold


import numpy as np
from scipy.optimize import curve_fit


def calculate_gof(df, metric='duration'):
    """
    计算拟合优度 (R²)，支持 duration 和 velocity 两种类型。
    :param df: 包含 sac_amp 和 duration / peakv 的 DataFrame
    :param metric: 'duration' 或 'velocity'
    :return: 拟合参数, R²
    """
    if metric == 'duration':
        df = df[df['duration'] < 150].dropna(subset=['sac_amp', 'duration'])
        x = df['sac_amp'].values
        y = df['duration'].values
        fit_func = slope_func_duration  # 你应当定义这个函数
        if len(df) > 2:
            try:
                inliers_mask = detect_outliers_gmm(np.column_stack((x, y)))
                x, y = x[inliers_mask], y[inliers_mask]
            except Exception as e:
                print(f"GMM failed: {e}")
                return None, None
    elif metric == 'velocity':
        df = df[df['peakv'] < 1000].dropna(subset=['sac_amp', 'peakv'])
        x = df['sac_amp'].values
        y = df['peakv'].values
        fit_func = power_func_velocity  # 你应当定义这个函数
    else:
        raise ValueError("metric must be 'duration' or 'velocity'")

    if len(x) > 2:
        try:
            params, _ = curve_fit(fit_func, x, y)
            y_pred = fit_func(x, *params)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            return params, r_squared
        except Exception as e:
            print(f"{metric.title()} fitting failed: {e}")
            return None, None
    else:
        print(f"Insufficient data for {metric} fitting.")
        return None, None


if __name__ == "__main__":
    sz_meta_file = f'{sz_dir}/meta_data/meta_data_release.xlsx'
    sz_meta = pd.concat([pd.read_excel(sz_meta_file, sheet_name=f"batch_{i}") for i in [0, 1]])
    results = []  # 用于保存所有记录

    for device in ['eyelink', 'phone']:
        for batch in ['batch_0', 'batch_1']:
            event_folder = os.path.join(f'{sz_dir}/events', f'data_{device}', batch)
            for file_name in os.listdir(event_folder):
                if 'sac' in file_name:
                    _, id_str, task = file_name.replace('.csv', '').split('_')
                    if task != 'fv':
                        continue
                    print(file_name)
                    id_int = int(id_str)
                    data_file = os.path.join(event_folder, file_name)
                    df = pd.read_csv(data_file) if os.path.exists(data_file) else pd.DataFrame()
                    dur_para, dur_r2 = calculate_gof(df, metric='duration')
                    vel_para, vel_r2 = calculate_gof(df, metric='velocity')
                    # print(dur_para)
                    meta_row = sz_meta[sz_meta['id'] == id_int]
                    if not meta_row.empty:
                        phone_both = meta_row.iloc[0]['phone_both']
                        eye_both = meta_row.iloc[0]['eyelink_both']
                        sz = meta_row.iloc[0]['sz']
                    else:
                        phone_both = eye_both = sz = None  # 缺失处理

                    # 添加结果到列表
                    results.append({
                        'id': id_int,
                        # 'task': task,
                        'device': device,
                        'batch': batch,
                        'dur_r2': dur_r2,
                        'vel_r2': vel_r2,
                        'dur_para_0': dur_para[0],
                        'vel_para_0': vel_para[0],
                        'dur_para_1': dur_para[1],
                        'vel_para_1': vel_para[1],
                        'phone_both': phone_both,
                        'eyelink_both': eye_both,
                        'label': sz
                    })

    # 转换为 DataFrame 并保存
    main_seq_df = pd.DataFrame(results)
    save_dir = 'sac_main_seq'
    os.makedirs(save_dir, exist_ok=True)
    main_seq_df.to_csv(os.path.join(save_dir, 'saccade_gof_summary.csv'), index=False)
    print(f"Saved summary to {os.path.join(save_dir, 'saccade_gof_summary.csv')}")

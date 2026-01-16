"""
提取抑郁检测眼动特征 - 增强版
包含多层次、多维度的眼动特征
优化版：移除microsaccade_ratio和全局count特征，优化特征计算
"""
from __future__ import annotations

import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
import tqdm
from scipy import stats
from scipy.spatial.distance import pdist

warnings.filterwarnings('ignore')

rois = [
    (74, 102, 934, 746),  # 上ROI
    (72, 1164, 934, 746)  # 下ROI
]

image_list = [
    "depression_images/HL_25.jpg", "depression_images/LH_15.jpg", "depression_images/LH_08.jpg",
    "depression_images/LH_28.jpg", "depression_images/HL_09.jpg", "depression_images/HL_14.jpg",
    "depression_images/HL_05.jpg", "depression_images/HL_16.jpg", "depression_images/LH_09.jpg",
    "depression_images/LH_25.jpg", "depression_images/HL_08.jpg", "depression_images/HL_02.jpg",
    "depression_images/LH_16.jpg", "depression_images/HL_10.jpg", "depression_images/HL_22.jpg",
    "depression_images/HL_30.jpg", "depression_images/HL_21.jpg", "depression_images/LH_24.jpg",
    "depression_images/HL_15.jpg", "depression_images/LH_13.jpg", "depression_images/LH_03.jpg",
    "depression_images/LH_26.jpg", "depression_images/LH_05.jpg", "depression_images/LH_18.jpg"
]

im_info = []
for img_path in image_list:
    img_name = os.path.basename(img_path).replace('.jpg', '')
    im_type, im_num = img_name.split('_')
    im_info.append({
        'type': im_type,  # HL或LH
        'number': int(im_num),
        'top_is_H': im_type == 'HL',  # 上ROI是否为高唤醒
        'bottom_is_H': im_type == 'LH'  # 下ROI是否为高唤醒
    })

def is_in_roi(x, y, roi):
    roi_x, roi_y, roi_w, roi_h = roi
    return roi_x <= x <= roi_x + roi_w and roi_y <= y <= roi_y + roi_h

def calculate_direction(start_x, start_y, end_x, end_y):
    dx = end_x - start_x
    dy = end_y - start_y
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad) % 360
    return angle_deg


def calculate_roi_features(fixations, im_type):
    # 初始化DataFrames
    if len(fixations) == 0:
        H_fixations = pd.DataFrame(columns=fixations.columns)
        L_fixations = pd.DataFrame(columns=fixations.columns)
    else:
        H_rows = []
        L_rows = []

        for _, row in fixations.iterrows():
            x_pix = row['avg_x_pix']
            y_pix = row['avg_y_pix']

            in_top = is_in_roi(x_pix, y_pix, rois[0])
            in_bottom = is_in_roi(x_pix, y_pix, rois[1])

            if in_top:
                if im_type == 'HL':  # 上ROI为H
                    H_rows.append(row)
                else:  # LH，上ROI为L
                    L_rows.append(row)
            elif in_bottom:
                if im_type == 'HL':  # 下ROI为L
                    L_rows.append(row)
                else:  # LH，下ROI为H
                    H_rows.append(row)

        H_fixations = pd.DataFrame(H_rows) if H_rows else pd.DataFrame(columns=fixations.columns)
        L_fixations = pd.DataFrame(L_rows) if L_rows else pd.DataFrame(columns=fixations.columns)

    return H_fixations, L_fixations


def calculate_fixation_features(fixations, prefix=''):
    """计算注视点特征"""
    if len(fixations) == 0:
        # 返回NaN而不是0
        return {
            # 注意：这里移除了count特征
            f'{prefix}duration_mean': np.nan,
            f'{prefix}duration_std': np.nan,
            f'{prefix}duration_median': np.nan,
            f'{prefix}duration_cv': np.nan,
            f'{prefix}x_mean_deg': np.nan,
            f'{prefix}x_std_deg': np.nan,
            f'{prefix}y_mean_deg': np.nan,
            f'{prefix}y_std_deg': np.nan,
            f'{prefix}dispersion_deg': np.nan,
            f'{prefix}scanpath_length': np.nan,
            f'{prefix}area_coverage': np.nan,
            f'{prefix}entropy': np.nan,
            f'{prefix}transition_rate': np.nan,
            f'{prefix}spatial_concentration': np.nan
        }

    features = {}

    # 基础统计特征（移除count）
    features[f'{prefix}duration_mean'] = np.mean(fixations['duration'])
    features[f'{prefix}duration_std'] = np.std(fixations['duration'])
    features[f'{prefix}duration_median'] = np.median(fixations['duration'])
    if features[f'{prefix}duration_mean'] > 0:
        features[f'{prefix}duration_cv'] = features[f'{prefix}duration_std'] / features[f'{prefix}duration_mean']
    else:
        features[f'{prefix}duration_cv'] = np.nan

    # 空间特征
    features[f'{prefix}x_mean_deg'] = np.mean(fixations['avg_x_deg'])
    features[f'{prefix}x_std_deg'] = np.std(fixations['avg_x_deg'])
    features[f'{prefix}y_mean_deg'] = np.mean(fixations['avg_y_deg'])
    features[f'{prefix}y_std_deg'] = np.std(fixations['avg_y_deg'])

    # 散布度和空间分布特征
    if len(fixations) > 1:
        coords = fixations[['avg_x_deg', 'avg_y_deg']].values

        # 散布度（平均注视点间距离）
        distances = pdist(coords, metric='euclidean')
        features[f'{prefix}dispersion_deg'] = np.mean(distances) if len(distances) > 0 else np.nan

        # 扫描路径长度（连续注视点间的距离和）
        scanpath_length = 0
        for i in range(len(coords) - 1):
            scanpath_length += np.linalg.norm(coords[i + 1] - coords[i])
        features[f'{prefix}scanpath_length'] = scanpath_length

        # 区域覆盖率（注视点分布的矩形面积）
        x_range = np.max(coords[:, 0]) - np.min(coords[:, 0])
        y_range = np.max(coords[:, 1]) - np.min(coords[:, 1])
        features[f'{prefix}area_coverage'] = x_range * y_range if x_range > 0 and y_range > 0 else np.nan

        # 空间信息熵（注视点分布的规律性）
        # 使用2D直方图计算熵
        x_bins = np.linspace(np.min(coords[:, 0]), np.max(coords[:, 0]), 10)
        y_bins = np.linspace(np.min(coords[:, 1]), np.max(coords[:, 1]), 10)
        hist, _, _ = np.histogram2d(coords[:, 0], coords[:, 1], bins=[x_bins, y_bins])
        hist_flat = hist.flatten()
        hist_flat = hist_flat[hist_flat > 0] / np.sum(hist_flat)
        features[f'{prefix}entropy'] = -np.sum(hist_flat * np.log2(hist_flat))

        # 空间集中度（使用核密度估计的概念）
        center = np.mean(coords, axis=0)
        distances_to_center = np.linalg.norm(coords - center, axis=1)
        features[f'{prefix}spatial_concentration'] = 1.0 / (1.0 + np.mean(distances_to_center))

        # 转移速率（单位时间的注视点转移次数）
        total_duration = np.sum(fixations['duration'])
        features[f'{prefix}transition_rate'] = len(fixations) / total_duration if total_duration > 0 else np.nan
    else:
        features[f'{prefix}dispersion_deg'] = np.nan
        features[f'{prefix}scanpath_length'] = np.nan
        features[f'{prefix}area_coverage'] = np.nan
        features[f'{prefix}entropy'] = np.nan
        features[f'{prefix}spatial_concentration'] = np.nan
        features[f'{prefix}transition_rate'] = np.nan

    return features


def calculate_saccade_features(saccades, prefix=''):
    """计算眼跳特征（移除microsaccade_ratio和count）"""
    if len(saccades) == 0:
        return {
            # 注意：这里移除了count特征和microsaccade_ratio
            f'{prefix}amplitude_mean_deg': np.nan,
            f'{prefix}amplitude_std_deg': np.nan,
            f'{prefix}amplitude_median_deg': np.nan,
            f'{prefix}duration_mean': np.nan,
            f'{prefix}duration_std': np.nan,
            f'{prefix}peak_velocity_mean': np.nan,
            f'{prefix}peak_velocity_std': np.nan,
            f'{prefix}mean_velocity_mean': np.nan,
            f'{prefix}mean_velocity_std': np.nan,
            f'{prefix}main_sequence_slope': np.nan,
            f'{prefix}amplitude_velocity_ratio': np.nan,
            f'{prefix}direction_entropy': np.nan,
            f'{prefix}velocity_consistency': np.nan,
            f'{prefix}amplitude_symmetry': np.nan
        }

    features = {}

    # 基础统计特征（移除count）
    features[f'{prefix}amplitude_mean_deg'] = np.mean(saccades['sac_amp_deg'])
    features[f'{prefix}amplitude_std_deg'] = np.std(saccades['sac_amp_deg'])
    features[f'{prefix}amplitude_median_deg'] = np.median(saccades['sac_amp_deg'])
    features[f'{prefix}duration_mean'] = np.mean(saccades['duration'])
    features[f'{prefix}duration_std'] = np.std(saccades['duration'])

    # 计算峰值速度
    if 'peak_velocity' in saccades.columns:
        peak_velocities = saccades['peak_velocity'].values
        features[f'{prefix}peak_velocity_mean'] = np.mean(peak_velocities)
        features[f'{prefix}peak_velocity_std'] = np.std(peak_velocities)
    else:
        # 使用主序列关系估算峰值速度：V_peak = 400 + 10*A (deg/s)
        peak_velocities = 100 * np.sqrt(saccades['sac_amp_deg'].values)
        features[f'{prefix}peak_velocity_mean'] = np.mean(peak_velocities)
        features[f'{prefix}peak_velocity_std'] = np.std(peak_velocities)

    # 计算平均速度（幅值/持续时间）
    saccades['mean_velocity'] = saccades['sac_amp_deg'] / saccades['duration'] * 1000  # deg/s
    features[f'{prefix}mean_velocity_mean'] = np.mean(saccades['mean_velocity'])
    features[f'{prefix}mean_velocity_std'] = np.std(saccades['mean_velocity'])

    # 计算主序列斜率（log-log线性回归）
    if len(saccades) > 1:
        try:
            # 使用幅值和峰值速度的对数
            log_amplitudes = np.log(saccades['sac_amp_deg'].values + 1e-10)
            log_velocities = np.log(peak_velocities + 1e-10)

            # 移除异常值（超过3个标准差）
            z_scores = np.abs((log_velocities - np.mean(log_velocities)) / (np.std(log_velocities) + 1e-10))
            valid_indices = z_scores < 3

            if np.sum(valid_indices) > 2:  # 至少需要3个点
                slope, _, _, _, _ = stats.linregress(
                    log_amplitudes[valid_indices],
                    log_velocities[valid_indices]
                )
                features[f'{prefix}main_sequence_slope'] = slope
            else:
                features[f'{prefix}main_sequence_slope'] = np.nan
        except Exception as e:
            features[f'{prefix}main_sequence_slope'] = np.nan
    else:
        features[f'{prefix}main_sequence_slope'] = np.nan

    # 幅速比（平均幅值/平均速度）
    if features[f'{prefix}mean_velocity_mean'] > 0:
        features[f'{prefix}amplitude_velocity_ratio'] = (
            features[f'{prefix}amplitude_mean_deg'] / features[f'{prefix}mean_velocity_mean']
        )
    else:
        features[f'{prefix}amplitude_velocity_ratio'] = np.nan

    # 计算方向熵（眼跳方向的规律性）
    if len(saccades) > 1:
        try:
            directions = []
            for _, row in saccades.iterrows():
                direction = calculate_direction(
                    row['start_x_deg'], row['start_y_deg'],
                    row['end_x_deg'], row['end_y_deg']
                )
                directions.append(direction)

            directions = np.array(directions)

            # 将方向分为8个扇形（0-360度）
            hist, _ = np.histogram(directions, bins=8, range=(0, 360))
            hist = hist / hist.sum()

            # 添加小值避免log(0)
            hist = hist + 1e-10

            # 计算熵
            entropy = -np.sum(hist * np.log2(hist))
            features[f'{prefix}direction_entropy'] = entropy

            # 速度一致性（速度变异系数）
            if features[f'{prefix}peak_velocity_mean'] > 0:
                features[f'{prefix}velocity_consistency'] = (
                    features[f'{prefix}peak_velocity_std'] / features[f'{prefix}peak_velocity_mean']
                )
            else:
                features[f'{prefix}velocity_consistency'] = np.nan

            # 幅值对称性（向左vs向右，向上vs向下）
            horizontal_amps = []
            vertical_amps = []
            for _, row in saccades.iterrows():
                dx = row['end_x_deg'] - row['start_x_deg']
                dy = row['end_y_deg'] - row['start_y_deg']
                amp = row['sac_amp_deg']

                if abs(dx) > abs(dy):  # 主要水平运动
                    horizontal_amps.append(amp if dx > 0 else -amp)
                else:  # 主要垂直运动
                    vertical_amps.append(amp if dy > 0 else -amp)

            if len(horizontal_amps) > 1:
                features[f'{prefix}amplitude_symmetry'] = np.abs(np.mean(horizontal_amps))
            elif len(vertical_amps) > 1:
                features[f'{prefix}amplitude_symmetry'] = np.abs(np.mean(vertical_amps))
            else:
                features[f'{prefix}amplitude_symmetry'] = np.nan

        except Exception as e:
            features[f'{prefix}direction_entropy'] = np.nan
            features[f'{prefix}velocity_consistency'] = np.nan
            features[f'{prefix}amplitude_symmetry'] = np.nan
    else:
        features[f'{prefix}direction_entropy'] = np.nan
        features[f'{prefix}velocity_consistency'] = np.nan
        features[f'{prefix}amplitude_symmetry'] = np.nan

    return features


def calculate_roi_interaction_features(H_fixations, L_fixations):
    """计算ROI交互特征"""
    features = {}

    H_count = len(H_fixations)
    L_count = len(L_fixations)
    total_count = H_count + L_count

    # ROI偏好特征
    if total_count > 0:
        features['roi_H_preference'] = H_count / total_count
        features['roi_L_preference'] = L_count / total_count
        features['roi_preference_ratio'] = H_count / L_count if L_count > 0 else np.nan
        features['roi_preference_diff'] = H_count - L_count
    else:
        features['roi_H_preference'] = np.nan
        features['roi_L_preference'] = np.nan
        features['roi_preference_ratio'] = np.nan
        features['roi_preference_diff'] = np.nan

    # ROI注视时间比例
    if total_count > 0:
        H_duration_total = np.sum(H_fixations['duration']) if H_count > 0 else 0
        L_duration_total = np.sum(L_fixations['duration']) if L_count > 0 else 0
        total_duration = H_duration_total + L_duration_total

        if total_duration > 0:
            features['roi_H_duration_ratio'] = H_duration_total / total_duration
            features['roi_L_duration_ratio'] = L_duration_total / total_duration
            features['roi_duration_diff'] = H_duration_total - L_duration_total
        else:
            features['roi_H_duration_ratio'] = np.nan
            features['roi_L_duration_ratio'] = np.nan
            features['roi_duration_diff'] = np.nan
    else:
        features['roi_H_duration_ratio'] = np.nan
        features['roi_L_duration_ratio'] = np.nan
        features['roi_duration_diff'] = np.nan

    return features


def calculate_temporal_features(fixations, trial_duration=4000):
    """计算时间动态特征"""
    if len(fixations) == 0:
        return {
            'fixation_rate': np.nan,
            'attention_onset': np.nan,
            'attention_decay': np.nan,
            'temporal_clustering': np.nan,
            'duration_pattern': np.nan
        }

    features = {}

    # 注视速率（单位时间的注视点次数）
    features['fixation_rate'] = len(fixations) / (trial_duration / 1000)  # 次/秒

    # 注意起始时间（第一次注视的onset）
    if 'onset' in fixations.columns and len(fixations) > 0:
        features['attention_onset'] = fixations['onset'].min()
    else:
        features['attention_onset'] = np.nan

    # 注意衰减（后期注视持续时间减少程度）
    if len(fixations) > 3:
        # 将注视点分为前后两半
        split_point = len(fixations) // 2
        first_half_mean = np.mean(fixations.iloc[:split_point]['duration'].values)
        second_half_mean = np.mean(fixations.iloc[split_point:]['duration'].values)

        if first_half_mean > 0:
            features['attention_decay'] = second_half_mean / first_half_mean
        else:
            features['attention_decay'] = np.nan
    else:
        features['attention_decay'] = np.nan

    # 时间聚类（注视间隔的规律性）
    if 'onset' in fixations.columns and len(fixations) > 1:
        onsets = fixations['onset'].values
        intervals = np.diff(np.sort(onsets))  # 注视间隔

        if np.mean(intervals) > 0:
            # 变异系数
            features['temporal_clustering'] = np.std(intervals) / np.mean(intervals)
        else:
            features['temporal_clustering'] = np.nan
    else:
        features['temporal_clustering'] = np.nan

    # 持续时间模式（检测持续时间的变化模式）
    if len(fixations) > 2:
        durations = fixations['duration'].values

        # 计算持续时间序列的自相关（滞后1）
        mean_duration = np.mean(durations)
        if mean_duration > 0:
            # 标准化
            norm_durations = (durations - mean_duration) / np.std(durations)

            # 计算滞后1的自相关
            if len(norm_durations) > 1:
                corr = np.corrcoef(norm_durations[:-1], norm_durations[1:])[0, 1]
                features['duration_pattern'] = corr if not np.isnan(corr) else 0
            else:
                features['duration_pattern'] = 0
        else:
            features['duration_pattern'] = 0
    else:
        features['duration_pattern'] = 0

    return features


def safe_nanmean(values):
    """安全的nanmean，如果结果仍然是nan则返回0"""
    result = np.nanmean(values)
    return 0 if np.isnan(result) else result


def safe_nanstd(values):
    """安全的nanstd，如果结果仍然是nan则返回0"""
    result = np.nanstd(values)
    return 0 if np.isnan(result) else result


def aggregate_trial_features_to_summary(features_dict):
    """聚合trial特征，使用nanmean和nanstd，并确保最终没有NaN"""
    # 存储每个trial的特征值
    trial_data = {}

    # 收集所有trial特征
    for key, value in features_dict.items():
        if key.startswith('trial_'):
            parts = key.split('_', 2)
            if len(parts) == 3:
                trial_idx = parts[1]
                feature_name = parts[2]

                if feature_name not in trial_data:
                    trial_data[feature_name] = []
                trial_data[feature_name].append(value)

    # 计算均值和标准差，使用安全的nan函数
    aggregated_features = {}
    for feature_name, values in trial_data.items():
        if values:  # 确保有数据
            values_array = np.array(values)

            # 使用安全的nan函数
            mean_val = safe_nanmean(values_array)
            std_val = safe_nanstd(values_array)

            aggregated_features[f'{feature_name}_mean'] = mean_val
            aggregated_features[f'{feature_name}_std'] = std_val

    return aggregated_features


def extract_all_features(subj_id, fixation_file, saccade_file, timestamps):
    """提取所有特征"""
    features = {'subj': subj_id}

    # 读取数据
    try:
        fixations = pd.read_csv(fixation_file)
    except:
        fixations = pd.DataFrame()

    try:
        saccades = pd.read_csv(saccade_file)
    except:
        saccades = pd.DataFrame()

    # 整体特征（移除全局count特征）
    global_fix_features = calculate_fixation_features(fixations, 'global_fix_')
    global_sac_features = calculate_saccade_features(saccades, 'global_sac_')

    # 合并整体特征，替换NaN为0
    for key, value in global_fix_features.items():
        features[key] = 0 if np.isnan(value) else value
    for key, value in global_sac_features.items():
        features[key] = 0 if np.isnan(value) else value

    # 分试次特征
    trial_features = {}
    img_show_ts = timestamps['normalShowTimeStampList']
    cross_show_ts = timestamps['normalFixationShowTimeStampList']
    cross_show_ts = np.array(cross_show_ts)
    img_show_ts = np.array(img_show_ts)
    img_show_ts -= cross_show_ts[0]
    cross_show_ts -= cross_show_ts[0]
    cross_show_ts = cross_show_ts / 1e6
    img_show_ts = img_show_ts / 1e6
    trial_time_pair = []
    for i in range(len(img_show_ts)):
        if i != len(img_show_ts) - 1:
            start_time = img_show_ts[i]
            end_time = cross_show_ts[i + 1]
        else:
            start_time = img_show_ts[i]
            end_time = img_show_ts[i] + 4000  # add 4000 ms
        trial_time_pair.append((start_time, end_time))

    for i, im_info_item in enumerate(im_info):
        trial_start, trial_end = trial_time_pair[i]

        # 提取试次数据
        trial_fixations = fixations.loc[
            (fixations['onset'] >= trial_start) &  # 使用onset更准确
            (fixations['onset'] <= trial_end)
            ] if len(fixations) > 0 else pd.DataFrame()

        # ROI分类
        H_fixations, L_fixations = calculate_roi_features(trial_fixations, im_info_item['type'])

        # 试次基础特征
        trial_prefix = f'trial_{i}_'

        # 试次整体的注视特征
        trial_fixation_features = calculate_fixation_features(trial_fixations, trial_prefix)
        trial_features.update(trial_fixation_features)

        # ROI特征
        trial_features.update(calculate_fixation_features(H_fixations, f'{trial_prefix}H_'))
        trial_features.update(calculate_fixation_features(L_fixations, f'{trial_prefix}L_'))

        # ROI交互特征
        roi_features = calculate_roi_interaction_features(H_fixations, L_fixations)
        for k, v in roi_features.items():
            trial_features[f'{trial_prefix}{k}'] = v

        # 时间动态特征
        time_features = calculate_temporal_features(trial_fixations, trial_duration=4000)
        for k, v in time_features.items():
            trial_features[f'{trial_prefix}{k}'] = v

    # 将试次特征合并到主特征字典
    features.update(trial_features)

    # 聚合trial特征为均值和标准差
    aggregated_trial_features = aggregate_trial_features_to_summary(features)
    features.update(aggregated_trial_features)

    # 移除原始的trial开头的特征
    features_to_remove = [key for key in features.keys() if key.startswith('trial_')]
    for key in features_to_remove:
        features.pop(key)

    # 计算跨试次的一致性特征
    if 'H_duration_mean_mean' in features and features.get('H_duration_mean_mean', 0) > 0:
        features['consistency_HL_ratio_cv'] = features['H_duration_mean_std'] / features['H_duration_mean_mean']
    else:
        features['consistency_HL_ratio_cv'] = 0

    # 确保所有特征都没有NaN值
    for key in list(features.keys()):
        if key != 'subj' and (features[key] is None or (isinstance(features[key], float) and np.isnan(features[key]))):
            features[key] = 0

    return features


def separate_features_by_type(features_df):
    """将特征分为三组：fix特征、sac特征和其他特征"""
    # 提取被试ID
    subj_id = features_df['subj']

    # 定义特征分类
    fix_features = []
    sac_features = []
    other_features = []

    for column in features_df.columns:
        if column == 'subj':
            continue
        elif any(keyword in column for keyword in
                 ['sac_', 'saccade', 'global_sac', 'amplitude', 'velocity', 'direction', 'symmetry', '_sac', 'sac_', 'main_sequence']):
            # 移除包含'microsaccade'的特征
            if 'microsaccade' not in column:
                sac_features.append(column)
        elif any(keyword in column for keyword in
                 ['fix_', 'fixation', 'global_fix']):
            fix_features.append(column)
        elif any(keyword in column for keyword in ['roi', 'preference', 'consistency', 'diff', 'ratio', 'gender', 'age']):
            other_features.append(column)
        else:
            other_features.append(column)

    # 创建三个DataFrame，确保每个DataFrame都有subj列
    fix_df = pd.DataFrame({'subj': subj_id})
    if fix_features:
        fix_df = pd.concat([fix_df, features_df[fix_features]], axis=1)

    sac_df = pd.DataFrame({'subj': subj_id})
    if sac_features:
        sac_df = pd.concat([sac_df, features_df[sac_features]], axis=1)

    other_df = pd.DataFrame({'subj': subj_id})
    if other_features:
        other_df = pd.concat([other_df, features_df[other_features]], axis=1)

    return fix_df, sac_df, other_df


def main():
    """主函数"""
    # 创建特征存储目录
    feature_dir = os.path.join(os.path.dirname(__file__), 'features')
    os.makedirs(feature_dir, exist_ok=True)

    # 事件文件夹
    ev_folder = os.path.join(os.getcwd(), 'events')

    # 收集所有被试
    subjects = set()
    event_files = os.listdir(ev_folder)

    for file in event_files:
        if file.endswith('.csv'):
            try:
                parts = file.split('.')[0].split('_')
                if len(parts) == 3:
                    subjects.add(parts[1])
            except:
                continue

    # 为每个被试提取特征
    all_features = []

    for subj in tqdm.tqdm(subjects):
        # 查找文件
        fix_file = None
        sac_file = None

        for file in event_files:
            if file.startswith(f'fix_{subj}_'):
                fix_file = os.path.join(ev_folder, file)
            elif file.startswith(f'sac_{subj}_'):
                sac_file = os.path.join(ev_folder, file)

        if fix_file and sac_file:
            try:
                timestamp_json = f"{os.path.dirname(os.path.abspath(__file__))}/raw_data/data/{subj}/*/fv_timestamps.json"
                timestamp_json = glob.glob(timestamp_json)[0]
                with open(timestamp_json, 'r') as f:
                    timestamps = json.load(f)
                info_json = f"{os.path.dirname(os.path.abspath(__file__))}/raw_data/data/{subj}/*/SubjectInformation.json"
                info_json = glob.glob(info_json)[0]
                with open(info_json, 'r') as f:
                    info = json.load(f)
                gender = 1 if info['subject_gender'] == 'male' else 0
                age = info['subject_age']
                features = extract_all_features(subj, fix_file, sac_file, timestamps)
                features['gender'] = gender
                features['age'] = age
                all_features.append(features)
            except Exception as e:
                print(f"处理被试 {subj} 时出错: {e}")
        else:
            print(f"被试 {subj} 缺少文件")

    # 转换为DataFrame并保存
    if all_features:
        df_features = pd.DataFrame(all_features)

        # 再次检查并确保没有NaN值
        df_features = df_features.fillna(0)

        # 分离特征
        fix_df, sac_df, other_df = separate_features_by_type(df_features)

        # 保存完整特征集
        output_file = os.path.join(feature_dir, 'depression_eyetracking_features.csv')
        df_features.to_csv(output_file, index=False)

        print(f"特征提取完成！")
        print(f"总特征数: {len(df_features.columns)}")
        print(f"总样本数: {len(df_features)}")
        print(f"Fix特征数: {len(fix_df.columns) - 1}")
        print(f"Sac特征数: {len(sac_df.columns) - 1}")
        print(f"其他特征数: {len(other_df.columns) - 1}")
        print(f"保存到: {output_file}")

        # 输出关键特征摘要
        key_features = [
            'global_sac_direction_entropy', 'global_sac_peak_velocity_mean',
            'global_sac_main_sequence_slope', 'global_fix_duration_mean',
            'roi_H_preference_mean', 'roi_L_preference_mean',
            'consistency_HL_ratio_cv', 'fixation_rate_mean',
            'attention_decay_mean', 'temporal_clustering_mean'
        ]

        # 过滤出实际存在的特征
        available_features = [f for f in key_features if f in df_features.columns]

        if available_features:
            print("\n关键特征统计:")
            print(df_features[available_features].describe().round(3))

            # 检查NaN值
            nan_count = df_features[available_features].isna().sum().sum()
            print(f"\n关键特征中NaN值数量: {nan_count}")

            # 显示前几个被试的关键特征
            print("\n前5个被试的关键特征:")
            print(df_features[['subj'] + available_features[:5]].head().round(3))
    else:
        print("未提取到任何特征数据")


if __name__ == "__main__":
    main()

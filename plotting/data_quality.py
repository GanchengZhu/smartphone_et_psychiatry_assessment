import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view  # 新增滑动窗口计算
from scipy.stats import ttest_ind

from depression_symptom_detection import dep_dir
from schizophrenia_detection import sz_dir


# ========================== 核心算法优化 ==========================
def sz_analyze_d3_per_position(cal_res, window_ms=175):
    DEVICE_X_DPI = 428 / 7.12
    DEVICE_Y_DPI = 926 / 15.406
    SCREEN_DISTANCE_FACTOR = 272.16
    DISTANCE_EXPONENT = -0.985

    # 向量化计算距离和PPD
    iris_mean = (cal_res['iris_pixel_x'] + cal_res['iris_pixel_y']) / 2
    cal_res['dist'] = SCREEN_DISTANCE_FACTOR * np.power(iris_mean, DISTANCE_EXPONENT)
    cal_res['ppd_x'] = DEVICE_X_DPI * cal_res['dist'] * np.pi / 180
    cal_res['ppd_y'] = DEVICE_Y_DPI * cal_res['dist'] * np.pi / 180

    grouped = cal_res.groupby(['x_gt', 'y_gt'])
    acc_list, precision_list = [], []

    for (_, _), group_df in grouped:
        timestamps = group_df['gaze_timestamp'].values
        if len(timestamps) < 2:
            continue

        sample_rate = 1e9 / np.mean(np.diff(timestamps))  # 向量化计算采样率
        win_size = max(1, int(sample_rate * window_ms / 1000))  # 确保最小窗口为1

        if len(group_df) < win_size:
            continue

        # 向量化转换到视觉角度
        x_est = group_df['x_estimated'].values / group_df['ppd_x'].values
        y_est = group_df['y_estimated'].values / group_df['ppd_y'].values
        x_gt = group_df['x_gt'].values / group_df['ppd_x'].values
        y_gt = group_df['y_gt'].values / group_df['ppd_y'].values

        # 使用滑动窗口批量处理
        # windows = sliding_window_view(np.arange(len(group_df)), win_size)
        # valid_mask = windows[:, -1] < len(group_df)  # 验证窗口边界

        ex_win = sliding_window_view(x_est, win_size)
        ey_win = sliding_window_view(y_est, win_size)
        gx_win = sliding_window_view(x_gt, win_size)
        gy_win = sliding_window_view(y_gt, win_size)

        # 批量计算DVA
        mean_ex = ex_win.mean(axis=1)
        mean_ey = ey_win.mean(axis=1)
        mean_gx = gx_win.mean(axis=1)
        mean_gy = gy_win.mean(axis=1)
        dva_values = np.hypot(mean_gx - mean_ex, mean_gy - mean_ey)

        # 批量计算精度（RMS）
        diff_ex = np.diff(ex_win, axis=1)
        diff_ey = np.diff(ey_win, axis=1)
        rms_values = np.sqrt(np.mean(diff_ex ** 2 + diff_ey ** 2, axis=1))

        # d3_values = dva_values ** 2 * rms_values ** 2
        std_values = np.sqrt(np.mean((ex_win - mean_ex[:, None]) ** 2 + (ey_win - mean_ey[:, None]) ** 2, axis=1))
        d3_values = dva_values ** 2 * std_values ** 2
        # 只处理有效窗口
        # if valid_mask.any():
        best_idx = np.argmin(d3_values)
        acc_list.append(dva_values[best_idx])
        precision_list.append(rms_values[best_idx])

    return acc_list, precision_list


# ========================== 文件处理优化 ==========================
def get_file_time(f):
    """优化文件名解析"""
    base = os.path.basename(f)
    parts = base.split('_')
    try:
        return datetime(int(parts[-6]), int(parts[-5]), int(parts[-4]),
                        int(parts[-3]), int(parts[-2]), int(parts[-1]))
    except (ValueError, IndexError):
        return datetime.min


def process_single_file(subj, task_name, csv_file):
    """处理单个CSV文件的优化版本"""
    try:
        # 读取时指定数据类型和空值标记
        dtype_map = {
            'gaze_timestamp': float,
            'iris_pixel_x': float,
            'iris_pixel_y': float,
            'x_estimated': float,
            'y_estimated': float,
            'x_gt': float,
            'y_gt': float
        }

        df = pd.read_csv(
            csv_file,
            usecols=list(dtype_map.keys()),
            dtype=dtype_map,
            na_values=['NA', 'N/A', '#N/A', '', '-1.#IND', 'NAN'],
        )

        # 检查数据有效性
        if df.empty:
            print(f"警告: {csv_file} 无有效数据")
            return None

        acc, prec = sz_analyze_d3_per_position(df)

        if not acc or not prec:
            return None

        return {
            'id': subj,
            'task_str': task_name,
            'file': csv_file,
            'mean_accuracy': np.nanmean(acc),
            'median_accuracy': np.nanmedian(acc),
            'p80_accuracy': np.nanquantile(acc, 0.8),
            'mean_precision': np.nanmean(prec),
            'median_precision': np.nanmedian(prec),
            'p80_precision': np.nanquantile(prec, 0.8),
        }
    except Exception as e:
        print(f"Error processing {csv_file}: {str(e)}")
        return None


# ========================== 主体流程优化 ==========================
def sz_process_subject_folder(args):
    base_path, subj, meta_df = args
    subj_folder = os.path.join(base_path, subj)
    task_folders = []
    subj_id = int(subj)

    for task in ['sp', 'fv', 'fs']:
        if not meta_df.loc[meta_df['id'] == subj_id, f'phone_both'].iloc[0]:
            continue

        latest_folder = max(
            glob.glob(os.path.join(subj_folder, f"calibration_{task}_*")),
            key=get_file_time,
            default=None
        )
        if latest_folder:
            task_folders.append((subj, task, latest_folder))

    results = []
    for subj, task_name, task_folder in task_folders:
        csv_file = os.path.join(task_folder, 'Optional(1)_Optional(5).csv')
        if os.path.exists(csv_file):
            result = process_single_file(subj, task_name, csv_file)
            if result:
                results.append(result)

    return results


def process_eyelink_files(meta_df, data_quality_dir, sz_dir):
    """眼动仪数据处理优化"""
    acc_file = os.path.join(data_quality_dir, "sz_eyelink_accuracy.csv")
    os.makedirs(data_quality_dir, exist_ok=True)

    with open(acc_file, 'w') as f_error:
        f_error.write('id,task_str,mean_accuracy\n')

        for batch in ['batch_0', 'batch_1']:
            base_path = os.path.join(f'{sz_dir}/data_eyelink_asc', batch)
            for asc_file in os.listdir(base_path):
                if 'fx_001' in asc_file:
                    continue
                if asc_file.startswith('.'):
                    continue

                try:
                    parts = asc_file.split('_')
                    subj_id, task = parts[1], parts[0]
                    file_path = os.path.join(base_path, asc_file)

                    if not meta_df.loc[meta_df['id'] == int(subj_id), 'eyelink_both'].iloc[0]:
                        continue

                    with open(file_path, 'r') as f:
                        for line in f:
                            if 'TRIALID' in line:
                                break
                            if '!CAL VALIDATION' in line and 'ABORTED' not in line:
                                error_val = line.split()[9]
                                f_error.write(f"{subj_id},{task},{error_val}\n")
                                break
                except Exception as e:
                    print(f"Error processing {asc_file}: {str(e)}")


def calculate_sample_rate_from_timestamps(timestamps_ns):
    diffs = np.diff(timestamps_ns)
    if len(diffs) == 0:
        return None
    mean_diff_ns = np.mean(diffs)
    sample_rate = 1e9 / mean_diff_ns  # 纳秒 → 秒，再取倒数得 Hz
    return sample_rate


# for depression dataset
def dep_analyze_d3_per_position(cal_res, window_ms=175):
    xdpi = 1080 / 6.98
    ydpi = 2249 / 15.4

    # 计算视距和 PPD
    cal_res = cal_res[cal_res['showGaze'] == 1].copy()
    cal_res.loc[:, 'dist'] = (cal_res['leftDistance'] + cal_res['rightDistance']) / 2
    cal_res.loc[:, 'ppd_x'] = xdpi / (1 / cal_res['dist'] / np.pi * 180)
    cal_res.loc[:, 'ppd_y'] = ydpi / (1 / cal_res['dist'] / np.pi * 180)

    grouped = cal_res.groupby(['gtX', 'gtY'])

    acc_list = []
    precision_list = []
    # print(len(grouped))
    for (_, _), group_df in grouped:
        timestamps_ns = group_df['timestamp'].values
        sample_rate = calculate_sample_rate_from_timestamps(timestamps_ns)
        if sample_rate is None:
            continue

        win_size = int(np.round(window_ms / (1000 / sample_rate)))
        if len(group_df) < win_size:
            continue

        # 估计值与真值转换为视觉角
        x_est = group_df['filteredX'].values / group_df['ppd_x'].values
        y_est = group_df['filteredY'].values / group_df['ppd_y'].values
        x_gt = group_df['gtX'].values / group_df['ppd_x'].values
        y_gt = group_df['gtY'].values / group_df['ppd_y'].values

        d3_list, acc_temp, prec_temp = [], [], []

        # for i in range(len(group_df) - win_size + 1):
        #     ex, ey = x_est[i:i + win_size], y_est[i:i + win_size]
        #     gx, gy = x_gt[i:i + win_size], y_gt[i:i + win_size]
        #     dva = np.hypot(np.mean(gx) - np.mean(ex), np.mean(gy) - np.mean(ey))
        #     std = np.sqrt(np.mean((ex - np.mean(ex)) ** 2 + (ey - np.mean(ey)) ** 2))
        #     d3 = dva ** 2 * std ** 2
        #     rms = np.sqrt(np.mean(np.diff(ex) ** 2 + np.diff(ey) ** 2))
        #     d3_list.append(d3)
        #     acc_temp.append(dva)
        #     prec_temp.append(rms)

        # if d3_list:
        #     best = np.argmin(d3_list)
        #     acc_list.append(acc_temp[best])
        #     precision_list.append(prec_temp[best])

        ex_win = sliding_window_view(x_est, win_size)
        ey_win = sliding_window_view(y_est, win_size)
        gx_win = sliding_window_view(x_gt, win_size)
        gy_win = sliding_window_view(y_gt, win_size)

        # 批量计算DVA
        mean_ex = ex_win.mean(axis=1)
        mean_ey = ey_win.mean(axis=1)
        mean_gx = gx_win.mean(axis=1)
        mean_gy = gy_win.mean(axis=1)
        dva_values = np.hypot(mean_gx - mean_ex, mean_gy - mean_ey)

        # 批量计算精度（RMS）
        diff_ex = np.diff(ex_win, axis=1)
        diff_ey = np.diff(ey_win, axis=1)
        rms_values = np.sqrt(np.mean(diff_ex ** 2 + diff_ey ** 2, axis=1))
        std_values = np.sqrt(np.mean((ex_win - mean_ex[:, None]) ** 2 + (ey_win - mean_ey[:, None]) ** 2, axis=1))

        d3_values = dva_values ** 2 * std_values ** 2
        best_idx = np.argmin(d3_values)
        acc_list.append(dva_values[best_idx])
        precision_list.append(rms_values[best_idx])

    return acc_list, precision_list


# for depression dataset
def get_latest_time_file(folder_path):
    """
    从指定文件夹中找出时间最晚的 txt 文件，文件名格式为：yyyy_MM_dd_HH_mm_ss.txt
    返回该文件的完整路径（绝对路径）。
    """
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    def is_valid_time_file(filename):
        try:
            datetime.strptime(os.path.splitext(filename)[0], "%Y_%m_%d_%H_%M_%S")
            return True
        except:
            return False

    # 获取所有合法的时间文件
    files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".txt") and is_valid_time_file(f)
    ]

    if not files:
        raise Exception(f"No file(s) found in {folder_path}")

    # 找出时间最晚的
    latest_file = max(
        files,
        key=lambda f: datetime.strptime(os.path.splitext(f)[0], "%Y_%m_%d_%H_%M_%S")
    )

    return os.path.abspath(os.path.join(folder_path, latest_file))


def dep_process_subject_folder(args):
    """
    'id', 'task_str', 'file', 'median', 'mean', 'p80'
    :param args:
    :return:
    """
    # print(args)
    base_path, subj_task = args
    subj_task_folder = os.path.join(base_path, subj_task)
    subj = subj_task.replace('_FVTask', '')

    calibration_file_path = get_latest_time_file(subj_task_folder)

    results = []

    try:
        df = pd.read_csv(calibration_file_path, na_values='NAN', engine='python')
        acc, prec = dep_analyze_d3_per_position(df)
        results.append({
            'id': subj,
            'task_str': 'fv',
            'file': calibration_file_path,
            'mean_accuracy': np.nanmean(acc),
            'median_accuracy': np.nanmedian(acc),
            'p80_accuracy': np.quantile(acc, 0.8),
            'mean_precision': np.nanmean(prec),
            'median_precision': np.nanmedian(prec),
            'p80_precision': np.quantile(prec, 0.8),
        })

    except Exception as e:
        print(f"{base_path} {subj_task} error: {e}")
    return results


# for depression dataset
if __name__ == "__main__":
    sz_meta_file = f'{sz_dir}/meta_data/meta_data_release.xlsx'
    data_quality_dir = os.path.join(os.path.dirname(__file__), 'data_quality')
    fig_dir = os.path.join(os.path.dirname(__file__), 'fig_2')

    sz_meta = pd.concat([pd.read_excel(sz_meta_file, sheet_name=f"batch_{i}") for i in [0, 1]])

    # 并行处理批次数据
    all_results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for data_source in ['batch_0', 'batch_1']:
            base_path = f'{sz_dir}/data_phone/{data_source}'
            subjects = [s for s in os.listdir(base_path) if not s.startswith('.')]

            futures = [
                executor.submit(sz_process_subject_folder, (base_path, subj, sz_meta))
                for subj in subjects
            ]

            for future in as_completed(futures):
                all_results.extend(future.result() or [])

    # 保存结果
    os.makedirs(data_quality_dir, exist_ok=True)
    sz_phone_df = pd.DataFrame(all_results)
    sz_phone_df.to_csv(os.path.join(data_quality_dir, "sz_phone_data_quality.csv"), index=False)

    # 处理眼动仪数据
    process_eyelink_files(sz_meta, data_quality_dir, sz_dir)

    base_path = f'{dep_dir}/raw_data/validation'
    # print(os.listdir(base_path))
    subjects = [s for s in os.listdir(base_path) if not s.startswith('.') and s.endswith('_FVTask')]
    # print(subjects)
    all_results = []

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(dep_process_subject_folder, (base_path, subj)) for subj in subjects]
        for future in as_completed(futures):
            result = future.result()  # list of dicts
            all_results.extend(result)

    dep_df = pd.DataFrame(all_results, columns=['id',
                                                'task_str',
                                                'file',
                                                'mean_accuracy',
                                                'median_accuracy',
                                                'p80_accuracy',
                                                'mean_precision',
                                                'median_precision',
                                                'p80_precision', ])
    dep_df['id'] = dep_df['id'].astype(str).str.lower()
    dep_df.to_csv(f"{data_quality_dir}/dep_phone_data_quality.csv", index=False)


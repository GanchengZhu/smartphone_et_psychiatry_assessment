# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view  # 新增滑动窗口计算
from scipy.ndimage import label

from schizophrenia_detection import sz_dir


def sz_analyze_d3(gaze_ts, gaze_x, gaze_y, gt_x, gt_y, event, window_ms=175):
    assert np.all(np.diff(gaze_ts) == 1)

    # event为[1,1,1,1,1,1,0,0,0,0,1,1,1,1]代表每个样本对应的眼动事件fixation(0)，saccade(1)
    # 假设event含有x段fixation，那么请把gaze_x对应的fixation段找出来吧。
    fixation_mask = (event == 0)
    # 使用 label 对连续的 True 区段做标记
    labeled_array, num_features = label(fixation_mask)
    x_segments = [gaze_x[labeled_array == i] for i in range(1, num_features + 1)]
    y_segments = [gaze_y[labeled_array == i] for i in range(1, num_features + 1)]

    acc_list = []
    pre_list = []
    for x_seg, y_seg in zip(x_segments, y_segments):
        len_x_seg = len(x_seg)
        if len_x_seg < window_ms:
            continue

        x_seg_win = sliding_window_view(x_seg, window_ms)
        y_seg_win = sliding_window_view(y_seg, window_ms)
        mean_ex = x_seg_win.mean(axis=1)
        mean_ey = y_seg_win.mean(axis=1)
        dva_values = np.hypot(mean_ex - gt_x, mean_ey - gt_y)

        # 批量计算精度（RMS）
        diff_ex = np.diff(x_seg_win, axis=1)
        diff_ey = np.diff(y_seg_win, axis=1)
        rms_values = np.sqrt(np.mean(diff_ex ** 2 + diff_ey ** 2, axis=1))
        std_values = np.sqrt(np.mean((x_seg_win - mean_ex[:, None]) ** 2 + (y_seg_win - mean_ey[:, None]) ** 2, axis=1))
        d3_values = dva_values ** 2 * std_values ** 2
        if np.all(np.isnan(d3_values)):
            continue
        best_idx = np.nanargmin(d3_values)
        acc_list.append(dva_values[best_idx])
        pre_list.append(rms_values[best_idx])

    return acc_list, pre_list


def safe_float(value):
    try:
        return float(value.strip().strip(','))
    except Exception as e:
        return np.nan


def read_gaze_data(filepath, eye_selected="R", data_recording="R", trial_index=1):
    """
    FV和FS使用，从 trial start 开始读取；FV 只取第4个trial
    """
    gaze_ts, gaze_x, gaze_y, ppd_x, ppd_y, event = [], [], [], [], [], []
    trial_counter = 0
    recording = False

    current_event = 1
    with open(filepath, 'r', encoding='gbk') as f:
        for line in f:

            line = line.strip()
            if "blank_screen" in line:
                break

            if "fixcross_onset" in line:
                trial_counter += 1
                recording = (trial_counter == trial_index)
                continue

            if not recording:
                continue

            if f'SFIX' in line:
                current_event = 0
            elif f'SSACC' in line:
                current_event = 1

            if not line or not line[0].isdigit():
                continue

            try:
                fields = [safe_float(x) for x in re.findall(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', line)]
                n = len(fields)

                if n == 13:
                    timestamp, lx, ly, _, rx, ry, *_ = fields
                    x, y = (rx, ry) if eye_selected == "R" else (lx, ly)
                    _ppdx, _ppdy = fields[-2], fields[-1]

                elif n == 8:
                    timestamp, x, y, *_ = fields
                    _ppdx, _ppdy = fields[-2], fields[-1]

                else:
                    timestamp = fields[0]
                    x = y = _ppdx = _ppdy = np.nan

                gaze_ts.append(timestamp)
                gaze_x.append(x)
                gaze_y.append(y)
                ppd_x.append(_ppdx)
                ppd_y.append(_ppdy)
                event.append(current_event)

            except Exception as e:
                print(f"[Parsing Error] {e}")
                # 可选：追加 NaN 填补异常行
                gaze_ts.append(timestamp)
                gaze_x.append(np.nan)
                gaze_y.append(np.nan)
                ppd_x.append(np.nan)
                ppd_y.append(np.nan)
                event.append(current_event)

    return [np.array(i) for i in [gaze_ts, gaze_x, gaze_y, ppd_x, ppd_y, event]]


def sz_process_subject(args):
    base_path, file_name, meta_df = args
    task, subj_id, _ = file_name.split('_')
    subj_id = int(subj_id)

    if not meta_df.loc[meta_df['id'] == subj_id, f'eyelink_both'].iloc[0]:
        return None
    else:
        gaze_ts, gaze_x, gaze_y, ppd_x, ppd_y, event = read_gaze_data(os.path.join(base_path, file_name))
        gaze_angle_x = gaze_x / ppd_x
        gaze_angle_y = gaze_y / ppd_y
        gt_x = 960.0 / np.nanmean(ppd_x)
        gt_y = 540.0 / np.nanmean(ppd_y)
        acc_list, pre_list = sz_analyze_d3(gaze_ts, gaze_angle_x, gaze_angle_y, gt_x, gt_y, event)
        return subj_id, task, meta_df.loc[meta_df['id'] == subj_id, f'sz'].iloc[0], np.nanmean(acc_list), np.nanmean(
            pre_list)


if __name__ == "__main__":
    sz_meta_file = f'{sz_dir}/meta_data/meta_data_release.xlsx'
    data_quality_dir = os.path.join(os.path.dirname(__file__), 'data_quality')

    sz_meta = pd.concat([pd.read_excel(sz_meta_file, sheet_name=f"batch_{i}") for i in [0, 1]])

    all_results = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for data_source in ['batch_0', 'batch_1']:
            base_path = f'{sz_dir}/data_eyelink_asc/{data_source}'
            data_files = [s for s in os.listdir(base_path) if not s.startswith('.')]
            data_files = [s for s in data_files if s.startswith('fx')]

            futures = [
                executor.submit(sz_process_subject, (base_path, data_path, sz_meta))
                for data_path in data_files
            ]

            for future in as_completed(futures):
                if future.result() is not None:
                    all_results.append(future.result())

    # 保存结果
    os.makedirs(data_quality_dir, exist_ok=True)
    sz_phone_df = pd.DataFrame(all_results, columns=['subj_id', 'task', 'sz', 'acc', 'pre'])
    sz_phone_df.to_csv(os.path.join(data_quality_dir, "sz_phone_fs_data_quality.csv"), index=False)

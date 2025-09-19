# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import os
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from schizophrenia_detection import sz_dir

# 设置Nature子刊风格
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['axes.linewidth'] = 1.0
rcParams['lines.linewidth'] = 0.3
rcParams['font.size'] = 8
rcParams['xtick.major.width'] = 1.0
rcParams['ytick.major.width'] = 1.0
rcParams['figure.dpi'] = 600
fig_w, fig_h = 1, 0.31

random.seed(42)
sub_id = random.randint(1, 188)
print(f"Randomly selected subject ID: {sub_id}")

SCN_W, SCN_H = 1920, 1080

# Nature子刊配色方案
COLORS = {
    'fix': '#1F77B4',  # 蓝色
    'sac': '#D62728',  # 红色
    'blk': '#FFFFFF',  # 白色背景
}


def safe_float(value):
    try:
        return float(value.strip().strip(','))
    except:
        return np.nan


def read_gaze_data(filepath, eye_selected="R", data_recording="R", trial_index=4):
    """
    FV和FS使用，从 trial start 开始读取；FV 只取第4个trial
    """
    gaze_x, gaze_y, event = [], [], []
    trial_counter = 0
    recording = False

    current_event = 1
    with open(filepath, 'r', encoding='gbk') as f:
        for line in f:
            line = line.strip()
            if "fixcross_onset" in line or "image_onset" in line.lower():
                trial_counter += 1
                recording = (trial_counter == trial_index)
                continue

            if not recording:
                continue

            if f'SFIX' in line:
                current_event = 0
            elif f'SSACC' in line:
                current_event = 1

            parts = line.strip().split()
            if not parts or not parts[0][0].isdigit():
                continue

            try:
                if data_recording == "BI":
                    fields = [safe_float(val) for val in parts[:13]]
                    if len(fields) < 13:
                        continue
                    _, lx, ly, _, rx, ry, *_ = fields
                    x, y = (rx, ry) if eye_selected == "R" else (lx, ly)
                else:
                    fields = [safe_float(val) for val in parts[:8]]
                    if len(fields) < 3:
                        continue
                    _, x, y = fields[:3]

                if not np.isnan(x) and not np.isnan(y):
                    gaze_x.append(x)
                    gaze_y.append(y)
                    event.append(current_event)
            except Exception:
                continue

    return np.array(gaze_x), np.array(gaze_y), np.array(event)


def read_sp_data(filepath, eye_selected="R", data_recording="R"):
    """
    SP使用，只从movement开始之后读取
    """
    gaze_x, gaze_y, tar_x, tar_y, event = [], [], [], [], []
    current_tar_x, current_tar_y = np.nan, np.nan
    recording = False

    current_event = 1
    with open(filepath, 'r', encoding='gbk') as f:
        for line in f:
            line = line.strip()

            if "movement" in line.lower():
                recording = True
                continue

            if not recording:
                continue

            if f'SFIX' in line:
                current_event = 0
            elif f'SSACC' in line:
                current_event = 1

            if "TARGET_POS" in line:
                try:
                    parts = line.strip().split()
                    _, _, _, _, _, x, y, *_ = parts
                    current_tar_x = safe_float(x.replace(",", ""))
                    current_tar_y = safe_float(y)
                except:
                    continue
                continue

            parts = line.strip().split()
            if not parts or not parts[0][0].isdigit():
                continue

            try:
                if data_recording == "BI":
                    fields = [safe_float(val) for val in parts[:13]]
                    _, lx, ly, _, rx, ry, *_ = fields
                    x, y = (rx, ry) if eye_selected == "R" else (lx, ly)
                else:
                    fields = [safe_float(val) for val in parts[:8]]
                    if len(fields) < 3:
                        continue
                    _, x, y = fields[:3]

                if not np.isnan(x) and not np.isnan(y) and not np.isnan(current_tar_x) and not np.isnan(current_tar_y):
                    gaze_x.append(x)
                    gaze_y.append(y)
                    tar_x.append(current_tar_x)
                    tar_y.append(current_tar_y)
                    event.append(current_event)
            except:
                continue

    return np.array(gaze_x), np.array(gaze_y), np.array(tar_x), np.array(tar_y), np.array(event)


def plot_gaze_timeseries(gaze_x, gaze_y, event, title=""):
    """
    绘制Gaze时间序列图：
    - fixation/saccade颜色分段
    - 添加图例
    - 去除背景色和刻度
    """
    time = np.arange(len(gaze_x))
    fig, axs = plt.subplots(2, 1, figsize=(fig_w, fig_h), sharex=True, facecolor='white')

    for i, (gaze, label) in enumerate(zip([gaze_x, gaze_y], ['gaze x', 'gaze y'])):
        # axs[i].set_ylabel(label, fontsize=6, rotation='vertical')
        # axs[i].set_ylabel(label, fontsize=4, rotation=0,)
        # axs[i].set_title(label)

        start_idx = 0
        while start_idx < len(gaze):
            end_idx = start_idx
            while end_idx + 1 < len(gaze) and event[end_idx + 1] == event[start_idx]:
                end_idx += 1

            axs[i].plot(time[start_idx:end_idx + 1],
                        gaze[start_idx:end_idx + 1],
                        color=COLORS['fix'] if event[start_idx] == 0 else COLORS['sac'],
                        label='Fixation' if event[start_idx] == 0 else 'Saccade')

            start_idx = end_idx + 1

        # 去掉刻度线、背景
        axs[i].tick_params(axis='both', which='both', bottom=False, top=False,
                           left=False, right=False, labelbottom=False, labelleft=False)
        axs[i].set_facecolor('white')
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['bottom'].set_visible(False)

    # axs[-1].set_xlabel("Sample Time")
    # axs[-1].tick_params(labelbottom=True)

    # 添加图例，只显示一次
    handles = [
        plt.Line2D([0], [0], color=COLORS['fix'], lw=2, label='Fixation'),
        plt.Line2D([0], [0], color=COLORS['sac'], lw=2, label='Saccade'),
    ]
    # axs[0].legend(handles=handles, frameon=False, fontsize=24)

    # fig.suptitle(title)
    # plt.tight_layout()
    plt.savefig(f"{save_dir}/{title[:2]}_event.jpg")
    # plt.show()


save_dir = os.path.join(os.path.dirname(__file__), 'fig_1')
os.makedirs(save_dir, exist_ok=True)

# FV
fv_path = f"{sz_dir}/data_eyelink_asc/batch_0/fv_{sub_id}_1.asc"
if os.path.exists(fv_path):
    gaze_x, gaze_y, event = read_gaze_data(fv_path, trial_index=15)
    plot_gaze_timeseries(gaze_x, gaze_y, event, title="FV Task Gaze Time Series")
else:
    print(f"File not found: {fv_path}")

# FS
fs_path = f"{sz_dir}/data_eyelink_asc/batch_0/fx_{sub_id}_1.asc"
if os.path.exists(fs_path):
    gaze_x, gaze_y, event = read_gaze_data(fs_path, trial_index=1)
    plot_gaze_timeseries(gaze_x, gaze_y, event, title="FS Task Gaze Time Series")
else:
    print(f"File not found: {fs_path}")

# SP
sp_path = f"{sz_dir}/data_eyelink_asc/batch_0/ps_{sub_id}_1.asc"
if os.path.exists(sp_path):
    gaze_x, gaze_y, tar_x, tar_y, event = read_sp_data(sp_path)
    plot_gaze_timeseries(gaze_x, gaze_y, event, title="SP Task Gaze Time Series")
else:
    print(f"File not found: {sp_path}")

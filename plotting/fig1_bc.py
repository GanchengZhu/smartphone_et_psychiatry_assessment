# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com
import glob
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, gridspec

from depression_symptom_detection import dep_dir
from schizophrenia_detection import sz_dir


def dep_extract_trial_samples(subj_id, trial_id):
    """
    Load depression data for smartphone eye tracking folder.
    :param subj_id:
    :param trial_id:
    :return:
    """
    data_pattern_dir = f"{dep_dir}/raw_data/data/{subj_id}*/FVTask*"
    data_dir = glob.glob(data_pattern_dir)[0]

    et_path = os.path.join(data_dir, 'MobileEyeTrackingRecord.csv')
    et_json_path = os.path.join(data_dir, 'fv_timestamps.json')

    with open(et_json_path, 'r') as f:
        json_data = json.load(f)
    onset_value = json_data['normalShowTimeStampList'][trial_id]
    offset_value = json_data['normalFixationShowTimeStampList'][trial_id + 1]

    em_df = pd.read_csv(et_path)
    em_df = em_df[(em_df['timestamp'] < offset_value) & (em_df['timestamp'] > onset_value)]

    gaze_x = em_df['filteredX']
    gaze_y = em_df['filteredY']

    return np.column_stack([gaze_x, gaze_y])


def extract_samples_from_asc(file, experiment_type='sp', trial_id=None):
    """
    Extract eye movement samples from ASC file (supports multiple experiment paradigms)

    Parameters:
        file: Path to ASC file (str)
        experiment_type: Experiment type (str), options:
            'sp' - Smooth pursuit experiment
            'fv' - Free viewing experiment
            'fx' - Fixation experiment
        trial_id: Integer, required only for free viewing (fv), specifies trial number

    Returns:
        numpy array containing (x,y) coordinates with shape (n_samples, 2)
        Returns empty array (0, 2) if no valid data found
    """

    # Read all lines from ASC file
    with open(file, 'r') as asc:
        lines = asc.readlines()

    # Initialize start and end line markers
    start_line = None  # Line number where data begins
    end_line = None  # Line number where data ends

    # Determine data range based on experiment type
    if experiment_type == 'ps':
        # Smooth pursuit: Extract between movement_onset and blank_screen
        for i, line in enumerate(lines):
            if 'movement_onset' in line:
                start_line = i
            if 'blank_screen' in line:
                end_line = i
                break

    elif experiment_type == 'fv':
        # Free viewing: Extract specific trial between TRIALID markers
        if trial_id is None:
            raise ValueError("trial_id must be specified for free viewing experiment")

        # Find trial boundaries
        for i, line in enumerate(lines):
            if f'TRIALID {trial_id}' in line:
                start_line = i
            if f'TRIALID {trial_id + 1}' in line:
                end_line = i
                break

        # Handle edge cases
        if start_line is None:
            raise ValueError(f"TRIALID {trial_id} not found")
        if end_line is None:
            end_line = len(lines)  # Use EOF if no next trial

        # Find actual start point (image_onset within trial)
        for i in range(start_line, end_line):
            if 'image_onset' in lines[i]:
                start_line = i
                break

    elif experiment_type == 'fx':
        # Fixation: Extract between fixcross_onset and blank_screen
        for i, line in enumerate(lines):
            if 'fixcross_onset' in line:
                start_line = i
            if 'blank_screen' in line:
                end_line = i
                break
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    # Return empty array if no valid range found
    if start_line is None or end_line is None:
        return np.empty((0, 2))

    # Process sample lines in the determined range
    samples = []
    for line in lines[start_line:end_line]:
        # Extract all numerical values from line
        tmp_data = [float(x) for x in re.findall(r'-?\d+\.?\d*', line)]

        # Only process lines starting with digits (sample lines)
        if re.search('^\d', line):
            # Handle different ASC formats:
            if len(tmp_data) == 13:  # Extended format
                samples.append([tmp_data[4], tmp_data[5]])  # x,y positions
            elif len(tmp_data) == 8:  # Basic format
                samples.append([tmp_data[1], tmp_data[2]])
            else:  # Invalid format
                samples.append([np.nan, np.nan])

    return np.array(samples)


def extract_trial_samples(em_file, json_file, trial_id=15, task_name='fv'):
    """
    Extract eye movement samples for smartphone eye tracking experiments.

    Parameters:
        em_file (str): Path to eye movement CSV data file
        json_file (str): Path to experiment metadata JSON file
        trial_id (int): Trial number to extract (default=15)
        task_name (str): Experiment task type: 'fv'(free viewing),
                        'fs'(fixation stability), or 'sp'(smooth pursuit)

    Returns:
        numpy.ndarray: 2D array of (x,y) gaze coordinates during specified trial

    Raises:
        ValueError: If invalid task name is provided
    """
    # Read eye movement data and experiment metadata
    em_df = pd.read_csv(em_file, engine='python', na_values='NAN')
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # Determine time window based on task type
    if task_name == 'fv':
        # Free viewing: Extract between fixation onset and picture offset
        task_ts_start = np.int64(json_data['fixationOnsetList'][trial_id] / 1e6)
        task_ts_end = np.int64(json_data['pictureOffsetList'][trial_id] / 1e6)
    elif task_name == 'fs':
        # Fixation stability: Extract entire task duration
        task_ts_start = np.int64(json_data['preparationOffset'] / 1e6)
        task_ts_end = np.int64(json_data['distractorOffsetList'][-1] / 1e6)
    elif task_name == 'sp':
        # Smooth pursuit: Extract between movement onset and offset
        task_ts_start = np.int64(json_data['smoothPursuitOnset'] / 1e6)
        task_ts_end = np.int64(json_data['smoothPursuitOffset'] / 1e6)
    else:
        raise ValueError("Invalid task name")

    # Convert timestamps to milliseconds and filter data
    em_df['gt_ms'] = em_df['gaze_timestamp'] = np.int64(em_df['gaze_timestamp'] / 1e6)
    mask = (em_df['gt_ms'] > task_ts_start) & (em_df['gt_ms'] < task_ts_end)

    # Extract and return x,y coordinates
    x_estimated = em_df['x_estimated'][mask].values
    y_estimated = em_df['y_estimated'][mask].values
    return np.column_stack((x_estimated, y_estimated))


def get_unique_task_path(subject_dir, task_prefix):
    """
    Locate unique experiment data path for a given task.

    Parameters:
        subject_dir (str): Directory containing subject data
        task_prefix (str): Task name prefix (e.g., 'fv', 'fs', 'sp')

    Returns:
        tuple: (csv_path, json_path) paths to eye movement data and metadata

    Raises:
        FileNotFoundError: If no/missing task directory or files
        RuntimeError: If multiple matching task directories exist
    """
    # Find all directories matching task pattern
    task_dirs = sorted(glob.glob(os.path.join(subject_dir, f"{task_prefix}_*")))

    # Validate directory existence
    if len(task_dirs) == 0:
        raise FileNotFoundError(f"No {task_prefix} directory found in {subject_dir}")
    elif len(task_dirs) > 1:
        raise RuntimeError(f"Multiple {task_prefix} directories found in {subject_dir}: {task_dirs}")

    # Verify required files exist
    task_dir = task_dirs[0]
    csv_path = os.path.join(task_dir, "EyeMovementData.csv")
    json_path = os.path.join(task_dir, "exp_info.json")
    if not (os.path.exists(csv_path) and os.path.exists(json_path)):
        raise FileNotFoundError(f"Missing CSV or JSON in {task_dir}")

    return csv_path, json_path


def plot_sp(
        gaze_normal: np.ndarray,
        gaze_patient: np.ndarray,
        device_name: str,
        axs: list
):
    """Plot smooth pursuit trajectories comparing normal vs patient data against theoretical curve."""
    colors = {'HC': '#1f77b4', 'SZ': '#d62728'}
    t = np.linspace(0, 24, 200)

    if device_name.lower() == 'eyelink':
        SCN_W, SCN_H = 1920, 1080
        params = {'amp_x': SCN_W * 0.123, 'amp_y': SCN_H * 0.4405, 'freq_x': 1 / 8, 'freq_y': 1 / 12}
        x_curve = params['amp_x'] * np.sin(2 * np.pi * params['freq_x'] * t) + 960
        y_curve = SCN_H - params['amp_y'] * np.sin(2 * np.pi * params['freq_y'] * t) - 540
        ppd_x = 470.97 / 10
        ppd_y = 469.31 / 10
        x_px_min = np.min(x_curve)
        y_px_min = np.min(y_curve)
        x_px_max = np.max(x_curve)
        y_px_max = np.max(y_curve)
        x_px_center = (x_px_max + x_px_min) / 2
        y_px_center = (y_px_max + y_px_min) / 2
        x_curve = (x_curve - x_px_center) / ppd_x
        y_curve = (y_curve - y_px_center) / ppd_y
        x_lim = (-7, 7)
        y_lim = (-12, 12)
    elif device_name.lower() == 'phone':
        params = {'amp_x': 342.4, 'amp_y': 740.8, 'shift_x': 42.80, 'shift_y': 92.60, 'freq_x': 1 / 8, 'freq_y': 1 / 12}
        x_curve = params['shift_x'] + (params['amp_x'] - params['amp_x'] * np.sin(2 * np.pi * params['freq_x'] * t)) / 2
        y_curve = params['shift_y'] + (params['amp_y'] - params['amp_y'] * np.sin(2 * np.pi * params['freq_y'] * t)) / 2
        x_px_min = np.min(x_curve)
        y_px_min = np.min(y_curve)
        x_px_max = np.max(x_curve)
        y_px_max = np.max(y_curve)
        ppd_x = 68.045 / 2
        ppd_y = 73.34 / 2
        x_px_center = (x_px_max + x_px_min) / 2
        y_px_center = (y_px_max + y_px_min) / 2
        x_curve = (x_curve - x_px_center) / ppd_x
        y_curve = (y_curve - y_px_center) / ppd_y
        x_lim = (-7, 7)
        y_lim = (-12, 12)

    else:
        raise ValueError(f"Unsupported device: {device_name}")

    for ax, gaze, label in zip(axs, [gaze_normal, gaze_patient], ['HC', 'SZ']):
        gaze[:, 0] = (gaze[:, 0] - x_px_center) / ppd_x
        gaze[:, 1] = (gaze[:, 1] - y_px_center) / ppd_y

        ax.plot(x_curve, y_curve, '--', color='gray', linewidth=1, alpha=0.5)
        if len(gaze) > 1:
            ax.plot(gaze[:, 0], gaze[:, 1], color=colors[label], linewidth=1)
            # ax.set_title(("Smartphone | " if device_name == 'phone' else "EyeLink | ") + label, fontsize=6)
            ax.set_title(label, fontsize=6)
        ax.set(xlim=x_lim, ylim=y_lim)
        ax.invert_yaxis()
        # 设置以 2° 为间隔的刻度标签，单位为°
        xticks = [-7, 0, 7]
        yticks = np.arange(np.floor(y_lim[0]), np.ceil(y_lim[1]) + 1, 6)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([f"{int(x)}°" for x in xticks], fontsize=6)
        ax.set_yticklabels([f"{int(y)}°" for y in yticks], fontsize=6)


def plot_fs(
        gaze_normal: np.ndarray,
        gaze_patient: np.ndarray,
        device_name: str,
        axs: list
):
    colors = {'HC': '#1f77b4', 'SZ': '#d62728'}

    if device_name.lower() == 'eyelink':
        center_x, center_y = 960, 540
        ppd_x = 470.97 / 10
        ppd_y = 469.31 / 10
        # x_lim = (760, 1160)
        # y_lim = (740, 340)
    elif device_name.lower() == 'phone':
        center_x, center_y = 214, 463
        ppd_x = 68.045 / 2
        ppd_y = 73.34 / 2
        # padding = 200
        # x_lim = (center_x - padding, center_x + padding)
        # y_lim = (center_y + padding, center_y - padding)
    else:
        raise ValueError(f"Unsupported device: {device_name}")

    for ax, gaze, label in zip(axs, [gaze_normal, gaze_patient], ['HC', 'SZ']):

        if len(gaze) > 1:
            gaze[:, 0] = (gaze[:, 0] - center_x) / ppd_x
            gaze[:, 1] = (gaze[:, 1] - center_y) / ppd_y
            ax.plot(gaze[:, 0], gaze[:, 1], color=colors[label], linewidth=0.5)
            text_center_x, text_center_y = -0.40, 0.5
            if device_name == 'phone':
                ax.set_title(label, fontsize=6)
                if label == 'HC':
                    ax.text(text_center_x, text_center_y, 'Smartphone', rotation='vertical', va='center', ha='right',
                            fontsize=6,
                            transform=ax.transAxes)
                else:
                    ax.text(text_center_x, text_center_y, "", transform=ax.transAxes)
            else:
                ax.set_title("", fontsize=6)
                if label == 'HC':
                    ax.text(text_center_x, text_center_y, "EyeLink", rotation='vertical', va='center', ha='right',
                            fontsize=6,
                            transform=ax.transAxes)
                else:
                    ax.text(text_center_x, text_center_y, "", transform=ax.transAxes)

        # ax.plot([-0.5, 0.5], [0, 0], 'k-', linewidth=1.0)
        # ax.plot([0, 0], [-0.5, 0.5], 'k-', linewidth=1.0)
        ax.text(0, 0, '+', va='center', ha='center',fontsize=10,)
        x_lim, y_lim = (-5, 5), (-5, 5)
        ax.set(xlim=x_lim, ylim=y_lim)
        ax.invert_yaxis()
        # 设置刻度为 ±5° 范围，单位为°
        xticks = np.arange(-5, 6, 5)
        yticks = np.arange(-5, 6, 5)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([f"{int(x)}°" for x in xticks], fontsize=6)
        ax.set_yticklabels([f"{int(y)}°" for y in yticks], fontsize=6)


def extract_and_plot_task_data(task, phone_dir_normal, phone_dir_patient, axs_dict, trial_id=None):
    info = task_info[task]

    try:
        em_normal, json_normal = get_unique_task_path(phone_dir_normal, task)
        em_patient, json_patient = get_unique_task_path(phone_dir_patient, task)
    except Exception as e:
        print(f"获取{task}任务路径错误: {e}")
        return

    eyelink_normal = extract_samples_from_asc(
        f"{sz_dir}/data_eyelink_asc/batch_0/{info['eyelink_suffix']}_{normal_id}_1.asc",
        info['eyelink_suffix'],
        trial_id=trial_id
    )
    eyelink_patient = extract_samples_from_asc(
        f"{sz_dir}/data_eyelink_asc/batch_0/{info['eyelink_suffix']}_{patient_id}_1.asc",
        info['eyelink_suffix'],
        trial_id=trial_id
    )

    phone_normal = extract_trial_samples(em_normal, json_normal, task_name=task, trial_id=trial_id)
    phone_patient = extract_trial_samples(em_patient, json_patient, task_name=task, trial_id=trial_id)

    # 绘制手机数据
    info['plot_func'](gaze_normal=phone_normal, gaze_patient=phone_patient, device_name='phone', axs=axs_dict['phone'])

    # 绘制眼动仪数据
    info['plot_func'](gaze_normal=eyelink_normal, gaze_patient=eyelink_patient, device_name='eyelink',
                      axs=axs_dict['eyelink'])


if __name__ == "__main__":
    # ============================================================
    # plot configuration
    # ============================================================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica']
    plt.rcParams['font.size'] = 6
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['lines.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['figure.dpi'] = 600

    # fs_fig_size = (2.1, 1)
    # sp_fig_size = (2.25, 1.70)
    # fv_fig_size = (2.10, 1.24)
    # merge_fig_size = (fv_fig_size[0] * 3, sp_fig_size[1] + fv_fig_size[1])

    # create output dir
    fig_save_dir = (Path(__file__).resolve().parent.resolve() / 'fig_1').resolve()
    fig_save_dir.mkdir(exist_ok=True)

    normal_id = '132'
    patient_id = '004'

    normal_root = f"{sz_dir}/data_phone/batch_0/{normal_id}"
    patient_root = f"{sz_dir}/data_phone/batch_0/{patient_id}"

    task_info = {
        'sp': {'eyelink_suffix': 'ps', 'plot_func': plot_sp},
        'fs': {'eyelink_suffix': 'fx', 'plot_func': plot_fs},
    }

    gs = gridspec.GridSpec(6, 6, )
    fig1 = plt.figure(figsize=(7.0, 2.0))

    # SP 图：一行两个（phone, eyelink）
    ax_fs_smartphone_hc = fig1.add_subplot(gs[0:3, 0])
    ax_fs_smartphone_sz = fig1.add_subplot(gs[0:3, 1])
    ax_fs_eyelink_hc = fig1.add_subplot(gs[3:6, 0])
    ax_fs_eyelink_sz = fig1.add_subplot(gs[3:6, 1])

    # FS 图：每种设备分别两行（HC, SZ）
    ax_sp_smartphone_hc = fig1.add_subplot(gs[0:6, 2])
    ax_sp_smartphone_sz = fig1.add_subplot(gs[0:6, 3])
    ax_sp_eyelink_hc = fig1.add_subplot(gs[0:6, 4])
    ax_sp_eyelink_sz = fig1.add_subplot(gs[0:6, 5])

    # extract_and_plot_task_data('sp', normal_root, patient_root)
    # extract_and_plot_task_data('fs', normal_root, patient_root)
    extract_and_plot_task_data('sp', normal_root, patient_root, axs_dict={
        'phone': [ax_sp_smartphone_hc, ax_sp_smartphone_sz],
        'eyelink': [ax_sp_eyelink_hc, ax_sp_eyelink_sz],
    })

    extract_and_plot_task_data('fs', normal_root, patient_root, axs_dict={
        'phone': [ax_fs_smartphone_hc, ax_fs_smartphone_sz],
        'eyelink': [ax_fs_eyelink_hc, ax_fs_eyelink_sz],
    })

    fig_save_dir = (Path(__file__).resolve().parent.resolve() / 'fig_1').resolve()
    fig_save_dir.mkdir(exist_ok=True)

    save_path = fig_save_dir / "fig1_sp_fs.tiff"
    fig1.tight_layout()
    fig1.savefig(save_path, dpi=600, bbox_inches='tight')

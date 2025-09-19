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
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, image as mpimg, gridspec

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


def plot_fv(
        gaze_normal: np.ndarray,
        gaze_patient: np.ndarray,
        device_name: str,
        trial_image: dict,
        axes: List,
        study_name: str = 'sz',

):
    """
    Generate free-viewing experiment heatmap comparison for both devices.
    Creates a 3-panel figure showing:
        1. Baseline stimulus images
        2. HC subject's gaze heatmap
        3. SZ's gaze heatmap
    """

    # Unified Gaussian kernel function (vectorized implementation)
    def gaussian_2d(width, std_x, height=None, std_y=None):
        """Generate 2D Gaussian kernel"""
        height = width if height is None else height
        std_y = std_x if std_y is None else std_y
        x0, y0 = width / 2, height / 2
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        return np.exp(-(((x - x0) ** 2) / (2 * std_x ** 2) + ((y - y0) ** 2) / (2 * std_y ** 2)))

    # Unified heatmap generation function
    def generate_heatmap(gaze, size, gwh=200, gstd=None):
        """Convert gaze points to smoothed heatmap"""
        gstd = gwh / 6 if gstd is None else gstd
        gauss = gaussian_2d(gwh, gstd)
        offset = gwh // 2
        H, W = int(size[1] + gwh), int(size[0] + gwh)
        heatmap = np.zeros((H, W), dtype=np.float32)

        for point in gaze:
            if len(point) < 2 or np.isnan(point[0]) or np.isnan(point[1]):
                continue
            x, y = point[0], point[1]
            xi = int(x + offset - gwh / 2)
            yi = int(y + offset - gwh / 2)

            if 0 <= xi < W - gwh and 0 <= yi < H - gwh:
                try:
                    heatmap[yi:yi + gwh, xi:xi + gwh] += gauss
                except:
                    pass

        cropped = heatmap[offset:offset + size[1], offset:offset + size[0]]
        valid = cropped[cropped > 0]
        threshold = np.mean(valid) if valid.size > 0 else 0
        cropped[cropped < threshold] = np.nan
        return cropped

    # Panel plotting function for both devices
    def plot_panel(ax, heatmap=None, title=None, overlay=True):
        """Plot image + optional heatmap with device-specific settings"""
        # Device-specific display settings
        if device_name.lower() == 'eyelink':
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_max, y_min)
            top_extent = [top_left, top_left + im_w, top_top + im_h, top_top]
            bot_extent = [bot_left, bot_left + im_w, bot_top + im_h, bot_top]
            heatmap_extent = [0, SCN_W, SCN_H, 0]
            overlay_extent = [0, SCN_W, SCN_H, 0]
            # heatmap_extent = [0, SCN_W, 0, SCN_H]
            # overlay_extent = [0, SCN_W, 0, SCN_H]
        else:  # phone
            ax.set_xlim(64, 364)
            ax.set_ylim(707, 119)
            top_extent = [64, 364, 344, 119]
            bot_extent = [64, 364, 707, 482]
            heatmap_extent = [0, SCN_W, SCN_H, 0]
            overlay_extent = [0, SCN_W, SCN_H, 0]
            # heatmap_extent = [0, SCN_W, 0, SCN_H]
            # overlay_extent = [0, SCN_W, 0, SCN_H]

            if study_name == 'dep':
                top_rect = [75, 204, im_w, im_h]
                bot_rect = [75, 1368, im_w, im_h]
                heatmap_extent = [0, SCN_W, SCN_H, 0]
                overlay_extent = [0, SCN_W, SCN_H, 0]
                ax.set_xlim(top_rect[0], top_rect[0] + top_rect[2])
                ax.set_ylim(bot_rect[1] + bot_rect[3], top_rect[1], )  # 注意：y 轴反转
                top_extent = [top_rect[0], top_rect[0] + top_rect[2],
                              top_rect[1] + top_rect[3], top_rect[1]]
                bot_extent = [
                    bot_rect[0], bot_rect[0] + bot_rect[2],
                                 bot_rect[1] + bot_rect[3], bot_rect[1]]

        ax.axis('off')
        if title: ax.set_title(title, fontsize=6)

        # Display images if found
        if os.path.exists(top_img_path):
            ax.imshow(mpimg.imread(top_img_path), extent=top_extent, zorder=1)
        if os.path.exists(bot_img_path):
            ax.imshow(mpimg.imread(bot_img_path), extent=bot_extent, zorder=2)

        # Add semi-transparent blue overlay
        if overlay:
            blue_overlay = np.zeros((SCN_H, SCN_W, 4), dtype=np.float32)
            blue_overlay[..., 2] = 1.0  # Blue channel
            blue_overlay[..., 3] = 0.2  # Alpha transparency
            ax.imshow(blue_overlay, extent=overlay_extent, zorder=2.5)

        # Add heatmap if provided
        if heatmap is not None:
            ax.imshow(heatmap, cmap="jet", alpha=0.6, extent=heatmap_extent, zorder=3)

    # Get root directory for images
    root_dir = Path(__file__).resolve().parent.parent.resolve()

    gaussianwh = 120
    gaussiansd = 20

    # Device-specific configurations
    if device_name.lower() == 'eyelink':
        # Screen and image parameters
        SCN_W, SCN_H = 1920, 1080
        im_w, im_h = 409, 306
        pos_top, pos_bot = (0, 237), (0, -237)

        # Calculate image boundaries
        def get_img_coords(center_x, center_y):
            left = SCN_W / 2 + center_x - im_w / 2
            top = SCN_H / 2 - center_y - im_h / 2
            return left, top

        top_left, top_top = get_img_coords(*pos_top)
        bot_left, bot_top = get_img_coords(*pos_bot)
        x_min = min(top_left, bot_left)
        x_max = max(top_left + im_w, bot_left + im_w)
        y_min = top_top
        y_max = bot_top + im_h

        # Set image paths
        eyelink_exp_code_dir = root_dir / 'eyelink_experiment_code'
        top_img_path = eyelink_exp_code_dir / 'images' / trial_image['top']
        bot_img_path = eyelink_exp_code_dir / 'images' / trial_image['bottom']

        fname = 'eyelink_fv.tiff'
    elif device_name.lower() == 'phone':
        # Screen parameters
        SCN_W, SCN_H = 428, 926
        # Set image paths
        img_dir = root_dir / 'eyelink_experiment_code' / 'images'
        top_img_path = img_dir / trial_image['top']
        bot_img_path = img_dir / trial_image['bottom']
        fname = 'smartphone_fv.tiff'
    else:
        raise ValueError(f"Unsupported device: {device_name}. Use 'eyelink' or 'phone'.")

    if study_name == "dep":
        SCN_W, SCN_H = 1080, 2316
        im_w, im_h = 930, 744
        current_dir = os.path.dirname(__file__)
        top_img_path = os.path.join(current_dir, "asset", trial_image['top'])
        bot_img_path = os.path.join(current_dir, "asset", trial_image['bottom'])
        fname = 'dep_smartphone_fv.tiff'

    # Generate heatmaps
    heat_normal = generate_heatmap(gaze_normal, (SCN_W, SCN_H), gaussianwh, gaussiansd)
    heat_patient = generate_heatmap(gaze_patient, (SCN_W, SCN_H), gaussianwh, gaussiansd)

    # Panel 1: Baseline image
    plot_panel(axes[0], title='Raw images', overlay=False)

    if study_name == 'dep':
        title = 'Non-symptom'
    else:
        title = 'HC'
    plot_panel(axes[1], heat_normal, title=title)

    # Panel 3: SZ heatmap
    if study_name == 'dep':
        title = 'Symptom'
    else:
        title = 'SZ'
    plot_panel(axes[2], heat_patient, title=title)

    if device_name == 'phone':
        device_name = 'smartphone'
    # Save output
    # fig.text(0.5, 0.22, ("Depression | " if study_name == 'dep' else 'Schizophrenia | ') + device_name.capitalize(),
    #          ha='center', fontsize=6)
    # # Save figure
    # plt.tight_layout()
    # plt.savefig(f'{fig_save_dir}/{fname}', bbox_inches='tight')
    # plt.close()


def extract_and_plot_task_data(task, phone_dir_normal, phone_dir_patient, trial_id=None):
    """
    提取并绘制指定任务的数据
    :param task: 任务标识 (sp, fs, fv)
    :param phone_dir_normal: 正常被试手机数据目录
    :param phone_dir_patient: 患者被试手机数据目录
    :param trial_id: 仅fv任务需要
    """
    info = task_info[task]

    # 获取手机数据路径
    try:
        em_normal, json_normal = get_unique_task_path(phone_dir_normal, task)
        em_patient, json_patient = get_unique_task_path(phone_dir_patient, task)
    except Exception as e:
        print(f"获取{task}任务路径错误: {e}")
        return

    # 提取眼动仪数据
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

    # 提取手机数据
    phone_normal = extract_trial_samples(em_normal, json_normal, task_name=task, trial_id=trial_id)
    phone_patient = extract_trial_samples(em_patient, json_patient, task_name=task, trial_id=trial_id)

    # 绘图
    plot_args = {
        'gaze_normal': eyelink_normal,
        'gaze_patient': eyelink_patient,
        'device_name': 'eyelink',
        'axes': [ax_eyelink_raw, ax_eyelink_hc, ax_eyelink_sz],
    }

    # 添加fv任务需要的额外参数
    if task == 'fv':
        plot_args['trial_image'] = images[trial_id]

    info['plot_func'](**plot_args)

    # 绘制手机数据
    plot_args.update({
        'gaze_normal': phone_normal,
        'gaze_patient': phone_patient,
        'device_name': 'phone',
        'axes': [ax_smartphone_raw, ax_smartphone_hc, ax_smartphone_sz],
    })

    info['plot_func'](**plot_args)


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

    gs = gridspec.GridSpec(1, 9, )
    fig1 = plt.figure(figsize=(7.0, 2.0))

    # SP 图：一行两个（phone, eyelink）
    ax_smartphone_raw = fig1.add_subplot(gs[0])
    ax_smartphone_hc = fig1.add_subplot(gs[1])
    ax_smartphone_sz = fig1.add_subplot(gs[2])
    ax_eyelink_raw = fig1.add_subplot(gs[3])
    ax_eyelink_hc = fig1.add_subplot(gs[4])
    ax_eyelink_sz = fig1.add_subplot(gs[5])

    # SP 图：一行两个（phone, eyelink）
    ax_dep_smartphone_raw = fig1.add_subplot(gs[6])
    ax_dep_smartphone_hc = fig1.add_subplot(gs[7])
    ax_dep_smartphone_sz = fig1.add_subplot(gs[8])

    # create output dir
    fig_save_dir = (Path(__file__).resolve().parent.resolve() / 'fig_1').resolve()
    fig_save_dir.mkdir(exist_ok=True)

    normal_id = '132'
    patient_id = '004'
    trial_id = 15

    images = [
        {"top": "im_15.jpg", "bottom": "im_16.jpg"}, {"top": "im_23.jpg", "bottom": "im_24.jpg"},
        {"top": "im_29.jpg", "bottom": "im_30.jpg"}, {"top": "im_13.jpg", "bottom": "im_14.jpg"},
        {"top": "im_1.jpg", "bottom": "im_2.jpg"}, {"top": "im_25.jpg", "bottom": "im_26.jpg"},
        {"top": "im_39.jpg", "bottom": "im_40.jpg"}, {"top": "im_17.jpg", "bottom": "im_18.jpg"},
        {"top": "im_11.jpg", "bottom": "im_12.jpg"}, {"top": "im_19.jpg", "bottom": "im_20.jpg"},
        {"top": "im_33.jpg", "bottom": "im_34.jpg"}, {"top": "im_3.jpg", "bottom": "im_4.jpg"},
        {"top": "im_21.jpg", "bottom": "im_22.jpg"}, {"top": "im_37.jpg", "bottom": "im_38.jpg"},
        {"top": "im_7.jpg", "bottom": "im_8.jpg"}, {"top": "im_27.jpg", "bottom": "im_28.jpg"},
        {"top": "im_9.jpg", "bottom": "im_10.jpg"}, {"top": "im_5.jpg", "bottom": "im_6.jpg"},
        {"top": "im_35.jpg", "bottom": "im_36.jpg"}, {"top": "im_31.jpg", "bottom": "im_32.jpg"}
    ]

    normal_root = f"{sz_dir}/data_phone/batch_0/{normal_id}"
    patient_root = f"{sz_dir}/data_phone/batch_0/{patient_id}"

    task_info = {
        'fv': {'eyelink_suffix': 'fv', 'plot_func': plot_fv, 'extra_args': 'trial_image=images[trial_id]'}
    }

    extract_and_plot_task_data('fv', normal_root, patient_root, trial_id=trial_id)

    dep_trial_stimuli_dict = {
        'top': 'top_4_H.jpg',
        'bottom': 'bot_4_L.jpg',
    }
    non_symptomatic_id = 'z78'
    symptomatic_id = 'z15'
    dep_trial_id = 4
    non_symptomatic_em = dep_extract_trial_samples(non_symptomatic_id, dep_trial_id)
    symptomatic_em = dep_extract_trial_samples(symptomatic_id, dep_trial_id)
    plot_fv(non_symptomatic_em, symptomatic_em, device_name='phone', trial_image=dep_trial_stimuli_dict,
            study_name='dep', axes=[ax_dep_smartphone_raw, ax_dep_smartphone_hc, ax_dep_smartphone_sz])

    save_path = fig_save_dir / "fig1_fv.tiff"
    fig1.tight_layout()
    fig1.savefig(save_path, dpi=600, bbox_inches='tight')
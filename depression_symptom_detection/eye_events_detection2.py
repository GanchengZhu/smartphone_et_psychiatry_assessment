from __future__ import annotations

import concurrent.futures
import json
import os

import numpy as np
import pandas as pd

from filter import HeuristicFilter

filter = HeuristicFilter(look_ahead=1)


def dispersion(positions: np.ndarray):
    """
    Compute the dispersion of a group of consecutive points in a 2D position time series.

    The dispersion is defined as the sum of the differences between
    the points' maximum and minimum x and y values

    Parameters
    ----------
    positions: array-like
        Continuous 2D position time series.

    Returns
    -------
    dispersion: float
        Dispersion of the group of points.
    """

    # print(np.nanmax(positions, axis=0),np.nanmin(positions, axis=0))
    return sum(np.nanmax(positions, axis=0) - np.nanmin(positions, axis=0))


def check_is_length_matching(**kwargs):
    """Check if two sequences are of equal length.

    Parameters
    ----------
    kwargs
        Keyword argument dictionary with 2 keyword arguments. Both values must be sequences.

    Raises
    ------
    ValueError
        If both sequences are of equal length , or if number of keyword arguments is not 2.
    """

    if len(kwargs) != 2:
        raise ValueError('there must be exactly two keyword arguments in kwargs')

    key_1, key_2 = (key for _, key in zip(range(2), kwargs.keys()))
    value_1 = kwargs[key_1]
    value_2 = kwargs[key_2]

    if not len(value_1) == len(value_2):
        raise ValueError(f'The sequences "{key_1}" and "{key_2}" must be of equal length.')


def idt(
        positions: np.ndarray,
        timesteps: np.ndarray,
        minimum_duration: int = 100,
        dispersion_threshold: float = 1.0,
        include_nan: bool = True,
        name: str = 'fixation'):
    """
    Fixation identification based on dispersion threshold.

    The algorithm identifies fixations by grouping consecutive points
    within a maximum separation (dispersion) threshold and a minimum duration threshold.
    The algorithm uses a moving window to check the dispersion of the points in the window.
    If the dispersion is below the threshold, the window represents a fixation,
    and the window is expanded until the dispersion is above threshold.

    The implementation and its default parameter values are based on the description and pseudocode
    from Salvucci and Goldberg :cite:p:`SalvucciGoldberg2000`.

    Parameters
    ----------
    positions: array-like, shape (N, 2)
        Continuous 2D position time series
    timesteps: array-like, shape (N, )
        Corresponding continuous 1D timestep time series. If None, sample based timesteps are
        assumed.
    minimum_duration: int
        Minimum fixation duration. The duration is specified in the units used in ``timesteps``.
         If ``timesteps`` is None, then ``minimum_duration`` is specified in numbers of samples.
    dispersion_threshold: float
        Threshold for dispersion for a group of consecutive samples to be identified as fixation
    include_nan: bool
        Indicator, whether we want to split events on missing/corrupt value (np.nan)
    name:
        Name for detected events in EventDataFrame.

    Returns
    -------
    pl.DataFrame
        A dataframe with detected fixations as rows.

    Raises
    ------
    TypeError
        If minimum_duration is not of type ``int`` or timesteps
    ValueError
        If positions is not shaped (N, 2)
        If dispersion_threshold is not greater than 0
        If duration_threshold is not greater than 0
    """
    positions = np.array(positions)

    timesteps = np.array(timesteps).flatten()

    # Check that timesteps are integers or are floats without a fractional part.
    timesteps_int = timesteps.astype(int)
    if np.any((timesteps - timesteps_int) != 0):
        raise TypeError('timesteps must be of type int')
    timesteps = timesteps_int

    check_is_length_matching(positions=positions, timesteps=timesteps)

    if dispersion_threshold <= 0:
        raise ValueError('dispersion_threshold must be greater than 0')
    if minimum_duration <= 0:
        raise ValueError('minimum_duration must be greater than 0')
    if not isinstance(minimum_duration, int):
        raise TypeError(
            'minimum_duration must be of type int'
            f' but is of type {type(minimum_duration)}',
        )

    onsets = []
    offsets = []
    start_x = []
    start_y = []
    end_x = []
    end_y = []
    avg_x = []
    avg_y = []
    duration = []

    # Infer minimum duration in number of samples.
    timesteps_diff = np.diff(timesteps)
    minimum_sample_duration = int(minimum_duration // np.mean(timesteps_diff))
    # print(np.mean(timesteps_diff))
    if minimum_sample_duration < 2:
        raise ValueError('minimum_duration must be longer than the equivalent of 2 samples')

    # Initialize window over first points to cover the duration threshold
    win_start = 0
    win_end = minimum_sample_duration

    while win_start < len(timesteps) and win_end <= len(timesteps):

        # Initialize window over first points to cover the duration threshold.
        # This automatically extends the window to the specified minimum event duration.
        win_end = max(win_start + minimum_sample_duration, win_end)
        win_end = min(win_end, len(timesteps))
        if win_end - win_start < minimum_sample_duration:
            break

        if dispersion(positions[win_start:win_end]) <= dispersion_threshold:
            # Add additional points to the window until dispersion > threshold.
            while dispersion(positions[win_start:win_end]) < dispersion_threshold:
                # break if we reach end of input data
                if win_end == len(timesteps):
                    break

                win_end += 1

            # Note a fixation at the centroid of the window points.
            onsets.append(timesteps[win_start])
            offsets.append(timesteps[win_end - 2])
            _sx, _sy = positions[win_start]
            _ex, _ey = positions[win_end - 2]
            _amp = np.hypot(_sx - _ex, _sy - _ey)
            start_x.append(_sx)
            start_y.append(_sy)
            end_x.append(_ex)
            end_y.append(_ey)
            avg_x.append(np.mean(positions[win_start:win_end - 2][:, 0]))
            avg_y.append(np.mean(positions[win_start:win_end - 2][:, 1]))
            duration.append(timesteps[win_end - 2] - timesteps[win_start] + 1)

            # Remove window points from points.
            # Initialize new window excluding the previous window
            win_start = win_end
        else:
            # Remove first point from points.
            # Move window start one step further without modifying window end.
            win_start += 1

    # Create proper flat numpy arrays.
    # print(onsets)
    # print(offsets)
    # onsets_arr = np.array(onsets).flatten()
    # offsets_arr = np.array(offsets).flatten()

    event_fix = pd.DataFrame({'onset': onsets,
                              'offset': offsets,
                              'start_x': start_x,
                              'start_y': start_y,
                              'end_x': end_x,
                              'end_y': end_y,
                              'avg_x': avg_x,
                              'avg_y': avg_y,
                              'duration': duration})

    event_sac = pd.DataFrame({'onset': offsets[:-1],
                              'offset': onsets[1:],
                              'start_x': end_x[:-1],
                              'start_y': end_y[:-1],
                              'end_x': end_x[1:],
                              'end_y': end_y[1:]})
    event_sac['sac_amp'] = np.hypot(event_sac['start_x'] - event_sac['end_x'],
                                    event_sac['start_y'] - event_sac['end_y'])
    event_sac['duration'] = event_sac['offset'] - event_sac['onset']

    return event_fix, event_sac


def process_folder(args):
    _subj, _folder, base_path, ev_folder, trace_folder = args
    try:
        if _folder == ".DS_Store":
            return None

        _task = _folder.split('_')[0]
        _fn = os.path.join(base_path, _subj, _folder, 'MobileEyeTrackingRecord.csv')
        print(f'Processing... {_subj}, {_folder} on thread {os.getpid()}')

        # 获取任务时间戳
        _task_fn = os.path.join(base_path, _subj, _folder, 'fv_timestamps.json')
        try:
            with open(_task_fn) as _tmp_json:
                _json = json.load(_tmp_json)
                if _task == 'FVTask':
                    _task_start = np.int64(_json['normalFixationShowTimeStampList'][0] / 1e6)
                    _task_end = np.int64((_json['normalFixationShowTimeStampList'][-1] + 5e9) / 1e6)
        except Exception as e:
            print(f"Error loading timestamps: {e}")
            return None

        # 处理数据
        filter = HeuristicFilter(look_ahead=1)  # 每个线程独立的过滤器
        _mix = pd.read_csv(_fn, engine='python', na_values='NAN')

        # filter the gaze signal with a Heuristic filter
        _coords = list(zip(_mix['filteredX'], _mix['filteredY']))
        _coords = [filter.filter_values(i) for i in _coords]  # heuristic filter
        x, y = zip(*_coords)
        _mix['gX'] = x
        _mix['gY'] = y

        # distance based on iris radius
        v_dist = (np.mean(_mix['leftDistance']) + np.mean(_mix['rightDistance'])) / 2.0

        # converting to degree of visual angles
        _mix['x_deg'] = _mix['gX'] / 1080 * 6.98 / v_dist / np.pi * 180
        _mix['y_deg'] = _mix['gY'] / 2249 * 15.4 / v_dist / np.pi * 180

        # gaze timestamps in ms
        _mix['gt_ms'] = np.int64((_mix['timestamp']) / 1000000)

        # event detection
        _fix, _sac = idt(_mix[['x_deg', 'y_deg']],
                         _mix['gt_ms'],
                         minimum_duration=120,
                         dispersion_threshold=1.5)

        # using the task start and end timestamps to filter the detected eye events
        # selecting rows based on condition
        _fix = _fix[(_fix['offset'] > _task_start) & (_fix['onset'] < _task_end)]
        _sac = _sac[(_sac['onset'] >= _task_start) & (_sac['offset'] <= _task_end)]

        # get the timestamps relative to task onsets
        _fix['onset'] = _fix['onset'] - _task_start
        _fix['offset'] = _fix['offset'] - _task_start
        _sac['onset'] = _sac['onset'] - _task_start
        _sac['offset'] = _sac['offset'] - _task_start

        # 保存结果
        _ev_sac_path = os.path.join(ev_folder, f'sac_{_subj}_{_task}.csv')
        _ev_fix_path = os.path.join(ev_folder, f'fix_{_subj}_{_task}.csv')
        _fix.to_csv(_ev_fix_path, index=False)
        _sac.to_csv(_ev_sac_path, index=False)

        return True
    except Exception as e:
        print(f"Error processing {_subj}/{_folder}: {e}")
        return None


if __name__ == "__main__":
    # 基础路径设置
    base_path = os.path.join(os.getcwd(), 'raw_data', 'data')
    ev_folder = os.path.join(os.getcwd(), 'events')
    trace_folder = os.path.join(os.getcwd(), 'gaze_trace')

    # 创建输出目录
    os.makedirs(ev_folder, exist_ok=True)
    os.makedirs(trace_folder, exist_ok=True)

    # 准备任务参数
    tasks = []
    _subjects = os.listdir(base_path)
    for _subj in _subjects:
        if _subj == ".DS_Store":
            continue
        subj_path = os.path.join(base_path, _subj)
        for _folder in os.listdir(subj_path):
            tasks.append((_subj, _folder, base_path, ev_folder, trace_folder))

    # 使用线程池处理任务
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(process_folder, tasks))

    print(f"Processing completed. {sum(r is not None for r in results)} tasks succeeded.")

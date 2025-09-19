"""
eye-event detection with the I-DT algorithm, code adapted from
https://github.com/aeye-lab/pymovements/blob/main/src/pymovements/events/detection/idt.py
"""
from __future__ import annotations

import json
import os

import matplotlib.pylab as plt
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

            tmp_candidates = [np.arange(win_start, win_end - 1, 1)]
            # Filter all candidates by minimum duration.
            tmp_candidates = [
                candidate for candidate in tmp_candidates
                if len(candidate) >= minimum_sample_duration
            ]

            for candidate in tmp_candidates:
                win_start = candidate[0]
                win_end = candidate[-1]

                # print('minimum_sample_duration', minimum_sample_duration, len(candidate),
                #       timesteps[win_end-1]-timesteps[win_start])

                # extract the samples that belong to a fixation
                _fix_smp = positions[win_start:win_end - 1]

                # Note a fixation at the centroid of the window points.
                onsets.append(timesteps[win_start])
                offsets.append(timesteps[win_end - 1])
                _sx, _sy = positions[win_start]
                _ex, _ey = positions[win_end - 1]
                _amp = np.hypot(_sx - _ex, _sy - _ey)
                start_x.append(_sx)
                start_y.append(_sy)
                end_x.append(_ex)
                end_y.append(_ey)
                avg_x.append(np.mean(_fix_smp[:, 0]))
                avg_y.append(np.mean(_fix_smp[:, 1]))
                duration.append(timesteps[win_end - 1] - timesteps[win_start] + 1)

                # dispersion within a fixation
                _dispersion = ((np.nanmax(_fix_smp[:, 0]) - np.nanmin(_fix_smp[:, 0])) +
                               (np.nanmax(_fix_smp[:, 1]) - np.nanmin(_fix_smp[:, 1])))
                # fix_dispersion.append(_dispersion)

                # scan path length (trajectory length)
                # spl = np.nansum(np.sqrt(np.diff(_fix_smp[:, 0]) ** 2 + np.diff(_fix_smp[:, 1]) ** 2))

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



# testing the algorithm with an input file
if __name__ == "__main__":

    show_plot = True
    show_vel = False

    # base folder
    base_path = os.path.join(os.getcwd(), 'raw_data', 'data')

    # create a folder to save the detected events
    ev_folder = os.path.join(os.getcwd(), 'events')
    if not os.path.exists(ev_folder):
        os.mkdir(ev_folder)

    # create a folder to save the gaze_trace "images"
    trace_folder = os.path.join(os.getcwd(), 'gaze_trace')
    if not os.path.exists(trace_folder):
        os.mkdir(trace_folder)

    _vel = []
    _acc = []
    _jrk = []
    _subjects = os.listdir(base_path)
    for _subj in _subjects:
        if _subj == ".DS_Store":
            continue
        # subject folder
        for _folder in os.listdir(os.path.join(base_path, _subj)):
            if _folder == ".DS_Store":
                continue
            else:
                _task = _folder.split('_')[
                    0]  # which task, could be fv (free viewing), sp (smooth pursuit),fs (fixation stability)
                _fn = os.path.join(base_path, _subj, _folder, 'MobileEyeTrackingRecord.csv')
                print('Processing...', _subj, _folder)

                # get the task timestamps, i.e., task start and end time
                _task_fn = os.path.join(base_path, _subj, _folder, 'fv_timestamps.json')
                try:
                    with open(_task_fn) as _tmp_json:
                        _json = json.load(_tmp_json)
                        # print(_json)
                        if _task == 'DotFollowing':
                            _task_start = np.int64(_json['normalCrossOnsetTimestamps'][0] / 1000000)
                            _task_end = np.int64(_json['normalProbeOffsetTimestamps'][-1] / 1000000)
                            continue
                        if _task == 'FaceMatrix':
                            _task_start = np.int64(_json['normalShowTimeStampList'][0] / 1000000)
                            _task_end = np.int64((_json['normalShowTimeStampList'][-1] + 5e9) / 1000000)
                            continue
                        if _task == 'FVTask':
                            _task_start = np.int64(_json['normalFixationShowTimeStampList'][0] / 1e6)
                            _task_end = np.int64((_json['normalFixationShowTimeStampList'][-1] + 5e9) / 1e6)

                        # print(_task, _task_end, _task_start, _task_end-_task_start)

                except Exception as e:
                    print(e)
                    continue

                # put the results of a single subject into pandas data frames
                # timestamp,trackingState,hasCalibrated,rawX,rawY,calibratedX,
                # calibratedY,filteredX,filteredY,leftDistance,rightDistance
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

                # detect events
                _fix, _sac = idt(_mix[['x_deg', 'y_deg']],
                                 _mix['gt_ms'],
                                 minimum_duration=120,  # 120
                                 dispersion_threshold=1.5  # 1.0
                                 )

                # using the task start and end timestamps to filter the detected eye events
                # selecting rows based on condition 
                _fix = _fix[(_fix['offset'] > _task_start) & (_fix['onset'] < _task_end)]
                _sac = _sac[(_sac['onset'] >= _task_start) & (_sac['offset'] <= _task_end)]

                # get the timestamps relative to task onsets
                _fix['onset'] = _fix['onset'] - _task_start
                _fix['offset'] = _fix['offset'] - _task_start
                _sac['onset'] = _sac['onset'] - _task_start
                _sac['offset'] = _sac['offset'] - _task_start

                # print out the detection results
                # print(_fix)
                # print(_sac)

                # save the detected events into CSV
                _ev_sac_path = os.path.join(ev_folder, 'sac_' + _subj + '_' + _task + '.csv')
                _ev_fix_path = os.path.join(ev_folder, 'fix_' + _subj + '_' + _task + '.csv')
                _fix.to_csv(_ev_fix_path, index=False)
                _sac.to_csv(_ev_sac_path, index=False)

                # data for plotting and smooth pursuit offset calculation
                _mix = _mix[(_mix['gt_ms'] >= _task_start) & (_mix['gt_ms'] < _task_end)]
                _mix['gt_ms'] = _mix['gt_ms'] - _task_start

                # check the sample intervals, as we need to assume equal intevals
                # for velocity, acceleration, and jerk calculation, when using np.gradient()
                # _timesteps_diff = np.diff(_mix['gt_ms'])
                # print(np.median(_timesteps_diff), np.std(_timesteps_diff))
                # plt.hist(_timesteps_diff)
                # plt.show()
                _mix['x_vel'] = np.gradient(_mix['x_deg'], _mix['gt_ms'] / 1000.)
                _mix['y_vel'] = np.gradient(_mix['y_deg'], _mix['gt_ms'] / 1000.)
                _mix['x_acc'] = np.gradient(_mix['x_vel'], _mix['gt_ms'] / 1000.)
                _mix['y_acc'] = np.gradient(_mix['y_vel'], _mix['gt_ms'] / 1000.)
                _mix['x_jrk'] = np.gradient(_mix['x_acc'], _mix['gt_ms'] / 1000.)
                _mix['y_jrk'] = np.gradient(_mix['y_acc'], _mix['gt_ms'] / 1000.)

                _mix['s_vel'] = np.hypot(_mix['x_vel'], _mix['y_vel'])
                _mix['s_acc'] = np.hypot(_mix['x_acc'], _mix['y_acc'])
                _mix['s_jrk'] = np.hypot(_mix['x_jrk'], _mix['y_jrk'])

                _vel.extend([np.max(_mix['s_vel']), np.min(_mix['s_vel'])])
                _acc.extend([np.max(_mix['s_acc']), np.min(_mix['s_acc'])])
                _jrk.extend([np.max(_mix['s_jrk']), np.min(_mix['s_jrk'])])

                # plot the gaze trace
                # show gaze traces with / without velocity traces
                if show_vel:
                    _mix.plot(x='gt_ms', y=['x_deg', 'y_deg', 'x_vel', 'y_vel'])
                else:
                    _mix.plot(x='gt_ms', y=['x_deg', 'y_deg'])

                # plot eye events (saccades and fixations)
                for _idx, _row in _fix.iterrows():
                    x0 = _row['onset']
                    x1 = _row['offset']
                    plt.axvspan(x0, x1, color='blue', alpha=0.1, lw=0)

                # show the plot
                if show_plot:
                    plt.savefig(os.path.join(trace_folder, f'{_folder}_{_subj}.png'))
                    # plt.show()

                # plt.scatter(_sac['sac_amp'], _sac['sac_amp']/(_sac['duration']/1000))
                # plt.show()

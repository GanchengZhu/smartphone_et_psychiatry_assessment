"""
eye-event detection with the I-DT algorithm, code adapted from 
https://github.com/aeye-lab/pymovements/blob/main/src/pymovements/events/detection/idt.py
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib.collections as mc
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
from scipy.signal import savgol_filter
from scipy.signal import butter, filtfilt
from filter import HeuristicFilter


def butter_lowpass(data, cutoff_freq, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def calculate_snr(samples):
    """Calculate the SNR for the smooth pursuit task based on the
    samples - the sample files
    """

    sp_tar_x = samples['smooth_pursuit_x'].to_numpy()
    sp_tar_y = samples['smooth_pursuit_y'].to_numpy()
    sp_gaze_x = samples['x_estimated'].to_numpy()
    sp_gaze_y = samples['y_estimated'].to_numpy()

    # noise (diff between signal and target)
    noise_x = sp_gaze_x - sp_tar_x
    signal_x = sp_tar_x
    noise_y = sp_gaze_y - sp_tar_y
    signal_y = sp_tar_y

    # signal energy（square）
    signal_energy_x = signal_x ** 2
    signal_energy_y = signal_y ** 2
    noise_energy_x = noise_x ** 2
    noise_energy_y = noise_y ** 2

    # 处理零值，避免在后续计算中产生不必要的nan
    signal_energy_x[signal_energy_x == 0] = np.nan
    signal_energy_y[signal_energy_y == 0] = np.nan
    noise_energy_x[noise_energy_x == 0] = np.nan
    noise_energy_y[noise_energy_y == 0] = np.nan

    # 计算最终的SNR，避免nan值干扰计算
    snr_x = 10 * np.log10(np.nansum(signal_energy_x) / np.nansum(noise_energy_x))
    snr_y = 10 * np.log10(np.nansum(signal_energy_y) / np.nansum(noise_energy_y))

    return snr_x, snr_y


def calculate_instantaneous_velocity(samples):
    """Using the 9-point velocity calculation methods seen in EyeLInk
    here we up-sample the gaze posiiton data to 250 Hz, and use a 5-point model instead
    samples -- the samples recorded in the smooth pursuit task
    """

    # up-sample to 250
    SR = 50

    # up-sample the data to 250 Hz
    _t = samples['gt_ms'].to_numpy()/1000
    tar_x = samples['smooth_pursuit_x'].to_numpy()
    tar_y = samples['smooth_pursuit_y'].to_numpy()
    gz_x = samples['x_estimated'].to_numpy()
    gz_y = samples['y_estimated'].to_numpy()

    # pixel per degree
    ppd_x = np.mean(samples['ppd_x'])
    ppd_y = np.mean(samples['ppd_y'])

    # 250 Hz time grid
    _tmp_ts = np.linspace(0, _t[-1], np.int64(SR*_t[-1]))  # 250 Hz time points

    # estimate the gaze and target velocity in the x direction
    _vx = []
    for _x in [tar_x,  gz_x]:
        _mask_x = ~np.isnan(_x)
        _time_x = _t[_mask_x]
        _clean_x = _x[_mask_x]
        cs_x = CubicSpline(_time_x, _clean_x)
        gz_x = cs_x(_tmp_ts)

        # x direction
        n_x = len(gz_x)
        vel_x = np.full(n_x, 0)
        for i in range(2, n_x - 2):
            numerator = gz_x[i + 2] + gz_x[i + 1] - gz_x[i - 1] - gz_x[i - 2]
            denominator = 6 * ppd_x
            vel_x[i] = SR * (numerator / denominator)
        vel_x = butter_lowpass(vel_x, cutoff_freq=0.5, sample_rate=250)
        _vx.append(vel_x)

    # estimate the gaze and target velocity in the y direction
    _vy = []
    for _y in [tar_y,  gz_y]:
        _mask_y = ~np.isnan(_y)
        _time_y = _t[_mask_y]
        _clean_y = _y[_mask_y]
        cs_y = CubicSpline(_time_y, _clean_y)
        gz_y = cs_y(_tmp_ts)

        # x direction
        n_y = len(gz_y)
        vel_y = np.full(n_y, 0)
        for i in range(2, n_y - 2):
            numerator = gz_y[i + 2] + gz_y[i + 1] - gz_y[i - 1] - gz_y[i - 2]
            denominator = 6 * ppd_y
            vel_y[i] = SR * (numerator / denominator)
        vel_y = butter_lowpass(vel_y, cutoff_freq=0.5, sample_rate=250)
        _vy.append(vel_y)

    # plt.plot(_tmp_ts,_vx[0])
    # plt.plot(_tmp_ts,_vy[0])
    # plt.plot(_tmp_ts,_vx[1])
    # plt.plot(_tmp_ts,_vy[1])
    # plt.ylim([-10, 10])
    # plt.show()

    # return velocity_tar_x, velocity_gz_x, velocity_tar_y, velocity_gz_y
    return _vx + _vy

def calculate_velocity(t, x, y, ppdx, ppdy, sample_rate=250, plot_results=False):
    """
    Calculate gaze velocity using upsampled 5-point differentiation.

    Parameters:
    -----------
    t : array-like
        Timestamps (in seconds)
    x, y : array-like
        Gaze/target positions (in pixels)
    ppdx, ppdy : array-like or float
        Pixels per degree conversion factors
    sample_rate : int, optional
        Target upsampling rate (default: 250 Hz)
    plot_results : bool, optional
        Whether to plot velocity results (default: False)

    Returns:
    --------
    vel_x, vel_y : ndarray
        Velocity in degrees per second for x and y directions
    """

    # Convert ppd to scalar means if they're arrays
    ppdx = np.nanmean(ppdx)
    ppdy = np.nanmean(ppdy)

    # zero the timestamp to prevent errors
    t = t - t[0]

    # Create high-resolution time grid
    end_time = t[-1]
    n_samples = int(np.round(sample_rate * end_time))
    time_hr = np.linspace(0, end_time, n_samples)

    def compute_hr_velocity(coord, time, ppd):
        """Helper function to compute velocity for one coordinate"""
        mask = ~np.isnan(coord)
        if np.sum(mask) < 5:  # Need at least 5 points for spline
            return np.full_like(time_hr, np.nan)

        # Interpolate to high-resolution grid
        cs = CubicSpline(time[mask], coord[mask])
        coord_hr = cs(time_hr)

        # 5-point central difference
        vel = np.full_like(coord_hr, np.nan)
        for i in range(2, len(coord_hr) - 2):
            numerator = coord_hr[i + 2] + coord_hr[i + 1] - coord_hr[i - 1] - coord_hr[i - 2]
            vel[i] = sample_rate * numerator / (6 * ppd)

        return vel

    # Calculate high-res velocities
    vel_x_hr = compute_hr_velocity(x, t, ppdx)
    vel_y_hr = compute_hr_velocity(y, t, ppdy)

    # Downsample to original timestamps
    def downsample(vel_hr, time_hr, original_t):
        """Downsample using linear interpolation"""
        valid_mask = ~np.isnan(vel_hr)
        if np.sum(valid_mask) < 2:
            return np.full_like(original_t, np.nan)
        return interp1d(
            time_hr[valid_mask],
            vel_hr[valid_mask],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )(original_t)

    vel_x = downsample(vel_x_hr, time_hr, t)
    vel_y = downsample(vel_y_hr, time_hr, t)

    # Optional plotting
    if plot_results:
        plt.figure(figsize=(12, 6))
        plt.plot(time_hr, vel_x_hr, alpha=0.5, label='X velocity (250Hz)')
        plt.plot(t, vel_x, 'o', markersize=4, label='X (downsampled)')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (deg/s)')
        plt.ylim([-100, 100])
        plt.legend()
        plt.grid(True)
        plt.title('Velocity Calculation Pipeline')
        plt.show()

    return vel_x, vel_y

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

def dispersion(positions:np.ndarray):
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
    return np.nansum(np.nanmax(positions, axis=0) - np.nanmin(positions, axis=0))


def idt(
        positions: np.ndarray,
        velocity: np.ndarray,
        timesteps: np.ndarray,
        minimum_duration: int = 120,
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
    from Salvucci and Goldberg :cite:p:`Salvucci_Goldberg2000`.

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
    fix_start_i = []
    fix_end_i = []
    duration = []
    fix_dispersion = []
    fix_spl = []

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
                win_start=candidate[0]
                win_end = candidate[-1]

                # print('minimum_sample_duration', minimum_sample_duration, len(candidate),
                #       timesteps[win_end-1]-timesteps[win_start])


                # extract the samples that belong to a fixation
                _fix_smp = positions[win_start:win_end-1]

                # Note a fixation at the centroid of the window points.
                onsets.append(timesteps[win_start])
                offsets.append(timesteps[win_end-1])
                _sx, _sy = positions[win_start]
                _ex, _ey = positions[win_end-1]
                _amp = np.hypot(_sx-_ex, _sy-_ey)
                start_x.append(_sx)
                start_y.append(_sy)
                end_x.append(_ex)
                end_y.append(_ey)
                avg_x.append(np.mean(_fix_smp[:,0]))
                avg_y.append(np.mean(_fix_smp[:,1]))
                duration.append(timesteps[win_end-1] - timesteps[win_start] + 1)

                # dispersion within a fixation
                _dispersion = ((np.nanmax(_fix_smp[:, 0]) - np.nanmin(_fix_smp[:, 0])) +
                               (np.nanmax(_fix_smp[:, 1]) - np.nanmin(_fix_smp[:, 1])))
                fix_dispersion.append(_dispersion)

                # scan path length (trajectory length)
                spl = np.nansum(np.sqrt(np.diff(_fix_smp[:, 0]) ** 2 + np.diff(_fix_smp[:, 1]) ** 2))
                fix_spl.append(spl)

                # start and end indices for saccade info extraction
                fix_start_i.append(win_start)
                fix_end_i.append(win_end-1)

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

    event_fix = pd.DataFrame({'onset':onsets,
                             'offset': offsets,
                             'start_x': start_x,
                             'start_y': start_y,
                             'end_x': end_x,
                             'end_y': end_y,
                             'avg_x': avg_x,
                             'avg_y': avg_y,
                             'duration': duration,
                              'dispersion': fix_dispersion,
                              'spl': fix_spl})
    event_fix['avg_vel'] = event_fix['spl']/event_fix['duration']/1000.


    event_sac = pd.DataFrame({'onset':offsets[:-1],
                             'offset': onsets[1:],
                             'start_x': end_x[:-1],
                             'start_y': end_y[:-1],
                             'end_x': end_x[1:],
                             'end_y': end_y[1:]})

    event_sac['sac_amp'] = np.hypot(event_sac['start_x'] - event_sac['end_x'],
                                    event_sac['start_y'] - event_sac['end_y'])
    event_sac['duration'] = event_sac['offset'] - event_sac['onset']

    # print(event_fix['onset'])
    # print(event_fix['offset'])

    # here to get the saccade peak velocity
    sac_start_i = fix_end_i[:-1]
    sac_end_i = fix_start_i[1:]
    vel = np.linalg.norm(velocity, axis=1)
    _sac_pv = []
    _sac_dispersion = []
    _sac_spl = []
    for i in range(len(sac_start_i)):
        _tmp_vel = vel[sac_start_i[i]-2:sac_end_i[i]+3]
        # print(_tmp_vel, np.nanmax(_tmp_vel))
        _sac_pv.append(np.nanmax(_tmp_vel))

        # extract the samples that belong to a saccade
        _sac_smp = positions[sac_start_i[i]:sac_end_i[i]+1]

        # dispersion within a saccade
        _dispersion = ((np.nanmax(_sac_smp[:, 0]) - np.nanmin(_sac_smp[:, 0])) +
                       (np.nanmax(_sac_smp[:, 1]) - np.nanmin(_sac_smp[:, 1])))
        _sac_dispersion.append(_dispersion)

        # scan path length (trajectory length)
        spl = np.nansum(np.sqrt(np.diff(_sac_smp[:, 0]) ** 2 + np.diff(_sac_smp[:, 1]) ** 2))
        _sac_spl.append(spl)

    event_sac['peakv'] = _sac_pv
    event_sac['dispersion'] = _sac_dispersion
    event_sac['spl'] = _sac_spl
    event_sac['avg_vel'] = event_sac['spl']/event_sac['duration']

    # plt.plot(timesteps, positions[:,0])
    # plt.plot(timesteps, positions[:,1])
    # plt.plot(timesteps, vel/10)
    #
    # # plot eye events (saccades and fixations)
    # for _idx, _row in event_fix.iterrows():
    #     x0 = _row['onset']
    #     x1 = _row['offset']
    #     plt.axvspan(x0, x1, color='blue', alpha=0.1, lw=0)
    #
    # for _idx, _row in event_sac.iterrows():
    #     x0 = _row['onset']
    #     x1 = _row['offset']
    #     plt.axvspan(x0, x1, color='red', alpha=0.1, lw=0)
    #
    # plt.show()

    return event_fix, event_sac


"""
Calculate the X position of the marker
"""
def tar_pos_x(time_elapsed: float, dist:float) -> float:
    amp_x = 465.0
    freq_x = 0.03125
    phase_x = 0.0
    size = 168
    xdpi = 370.70248/2.54
    _x_pos = (amp_x - size / 2.) * (np.sin(np.pi * 2. * freq_x * time_elapsed + phase_x) + 1.) + 75

    return _x_pos/xdpi/dist/np.pi*180

"""
Calculate the Y position of the marker
"""
def tar_pos_y(time_elapsed: float, dist:float) -> float:
    amp_y = 1049.5
    freq_y = 0.0416666666666667
    phase_y = 0.0
    size = 168
    ydpi = 372.31873/2.54
    _y_pos = (amp_y - size / 2.) * (np.sin(np.pi * 2. * freq_y * time_elapsed + phase_y) + 1.) + 75

    return _y_pos/ydpi/dist/np.pi*180

# testing the algorithm with an input file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', help='raw data path')
    args = parser.parse_args()

    # base folder
    base_path = os.path.join(os.getcwd(), 'data_phone', args.raw_data_path)

    # create a folder to save the detected events
    ev_folder = os.path.join(os.getcwd(), 'events', 'data_phone', args.raw_data_path)
    os.makedirs(ev_folder, exist_ok=True)

    # set up a heuristic filter to denoise the gaze data further for more robust event detection
    heuristic_filter = HeuristicFilter(look_ahead=2)

    _subjects = os.listdir(base_path)
    for _subj in _subjects:
        if _subj == ".DS_Store":
            continue
        # subject folder
        for _folder in os.listdir(os.path.join(base_path, _subj)):
            if _folder == ".DS_Store":
                continue
            if not ('calibration' in _folder):
                _task = _folder[:2]  # which task, could be fv (free viewing), sp (smooth pursuit),fs (fixation stability)
                _fn = os.path.join(base_path, _subj, _folder, 'EyeMovementData.csv')
                print('Processing...', _subj, _folder)

                # get the task timestamps, i.e., task start and end time
                _task_fn = os.path.join(base_path, _subj, _folder, 'exp_info.json')
                with open(_task_fn) as _tmp_json:
                    _json = json.load(_tmp_json)
                    if _task == 'fv':
                        _task_start = np.int64(_json['fixationOnsetList'][0]/1000000)
                        _task_end = np.int64(_json['pictureOffsetList'][-1]/1000000)
                    if _task == 'fs':
                        _task_start = np.int64(_json['preparationOffset']/1000000)
                        _task_end = np.int64(_json['distractorOffsetList'][-1]/1000000)
                    if _task == 'sp':
                        _task_start = np.int64(_json['smoothPursuitOnset']/1000000)
                        _task_end = np.int64(_json['smoothPursuitOffset']/1000000)

                    # print(_task, _task_end, _task_start, _task_end-_task_start)

                # put the results of a single subject into pandas data frames
                _mix = pd.read_csv(_fn,  engine='python', na_values='NAN')

                # converting to degree of visual angles
                xdpi = 428/7.12
                ydpi = 926/15.406
                _mix['dist'] = 272.16 * ((_mix['left_iris_pixel'] + _mix['right_iris_pixel'])/2)**(-0.985)
                v_dist = np.mean(_mix['dist'])
                _mix['ppd_x'] = xdpi/(1/_mix['dist']/np.pi*180)
                _mix['ppd_y'] = ydpi/(1/_mix['dist']/np.pi*180)

                # _mix['x_deg'] = _mix['x_estimated']/xdpi/v_dist/np.pi*180
                # _mix['y_deg'] = _mix['y_estimated']/ydpi/v_dist/np.pi*180

                # gaze timestamps in ms
                _mix['gt_ms'] = np.int64((_mix['gaze_timestamp'])/1000000)

                # apply the heuristic filter
                # _gaze_tuples_0 = list(zip(_mix['x_estimated'], _mix['y_estimated']))
                # _gaze_tuples_1 = []
                # for _gz in _gaze_tuples_0:
                #     _x, _y = heuristic_filter.filter_values(_gz)
                #     _gaze_tuples_1.append((_x, _y))
                # _gaze_tuples_2 = []
                # for _gz in _gaze_tuples_1:
                #     _x, _y = heuristic_filter.filter_values(_gz)
                #     _gaze_tuples_2.append((_x, _y))
                #
                # _df = pd.DataFrame(_gaze_tuples_2, columns=['gx', 'gy'])
                # _mix['gx'] = _df['gx']
                # _mix['gy'] = _df['gy']

                _mix['gx'] = _mix['x_estimated']
                _mix['gy'] = _mix['y_estimated']

                # calculate the velocity
                ts = _mix['gt_ms'].to_numpy()/1000  # timestamp in seconds
                gx = _mix['gx'].to_numpy()
                gy = _mix['gy'].to_numpy()

                # pixel per degree
                ppdx = _mix['ppd_x'].to_numpy()
                ppdy = _mix['ppd_y'].to_numpy()

                # estimation of instantaneous velocity
                _mix['x_vel'], _mix['y_vel'] = calculate_velocity(ts, gx, gy, ppdx, ppdy, plot_results=False)

                # gaze x and y in degree of visual angle
                _mix['x_deg'] = _mix['gx']/_mix['ppd_x']
                _mix['y_deg'] = _mix['gy']/_mix['ppd_y']

                # column names
                # #  x_raw       y_raw  x_filtered  y_filtered   gaze_timestamp  record_timestamp
                _fix, _sac = idt(
                    _mix[['x_deg', 'y_deg']],
                    _mix[['x_vel', 'y_vel']],
                    _mix['gt_ms'],
                    minimum_duration = 100, # 120
                    dispersion_threshold = 2.0  # 1.0
                    )

                # if _task=='fv':
                #     plt.scatter(_sac['sac_amp'], _sac['duration'])
                #     plt.ylim([0, 300])
                #     plt.xlim([0, 15])
                #     plt.show()
                #
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

                # in the smooth pursuit task, we calculate the x, y offsets
                if _task == 'sp':
                    # get the target location for each sample
                    # _mix['tar_x'] = tar_pos_x(_mix['gt_ms']/1000.)
                    # _mix['tar_y'] = tar_pos_y(_mix['gt_ms']/1000.)
                    # handle the column name change in V2
                    if 'x_gt_smooth_pursuit' in list(_mix.columns.values):
                        _mix['smooth_pursuit_x'] = _mix['x_gt_smooth_pursuit']
                    if 'y_gt_smooth_pursuit' in list(_mix.columns.values):
                        _mix['smooth_pursuit_y'] = _mix['y_gt_smooth_pursuit']

                    _mix['dist'] = 272.16 * ((_mix['left_iris_pixel'] + _mix['right_iris_pixel'])/2)**(-0.985)
                    _mix['ppd_x'] = xdpi/(1/_mix['dist']/np.pi*180)
                    _mix['ppd_y'] = ydpi/(1/_mix['dist']/np.pi*180)

                    _vtx, _vx, _vty, _vy = calculate_instantaneous_velocity(_mix)

                    v_dist = np.mean(_mix['dist'])

                    _mix['tar_x'] = _mix['smooth_pursuit_x']/xdpi/v_dist/np.pi*180
                    _mix['tar_y'] = _mix['smooth_pursuit_y']/ydpi/v_dist/np.pi*180

                    # print(np.max(_mix['tar_x']), np.max(_mix['tar_y']))

                    # calculate the x, y offsets
                    _mix['offset_x'] = np.abs(_mix['x_deg'] - _mix['tar_x'])
                    _mix['offset_y'] = np.abs(_mix['y_deg'] - _mix['tar_y'])
                    _sp = pd.DataFrame({'offset_x_mean': [np.mean(_mix['offset_x'])],
                                        'offset_x_p80': [_mix['offset_x'].quantile(0.5)],
                                        'offset_x_std': [np.std(_mix['offset_x'])],
                                        'offset_y_mean': [np.mean(_mix['offset_y'])],
                                        'offset_y_p80': [_mix['offset_y'].quantile(0.5)],
                                        'offset_y_std': [np.std(_mix['offset_y'])],
                                        'cor_x':_mix.corr()['tar_x']['x_deg'],
                                        'cor_y':_mix.corr()['tar_y']['y_deg']})

                    # calculate velocity gain
                    v_tar_x, v_gaze_x, v_tar_y, v_gaze_y = calculate_instantaneous_velocity(_mix)
                    # exclude the beginning and ending 0.5 sec. to reduce noise due to up-sampling and filtering
                    v_tar_x = v_tar_x[125:-125]
                    v_gaze_x = v_gaze_x[125:-125]
                    v_tar_y = v_tar_y[125:-125]
                    v_gaze_y = v_gaze_y[125:-125]

                    v_tar_x[v_tar_x == 0] = np.nan
                    v_tar_y[v_tar_y == 0] = np.nan
                    # print('velocity', len(v_tar_x), len(v_gaze_x))

                    _sp['gain_x'] = np.nanmedian(np.abs(v_gaze_x/v_tar_x))
                    _sp['gain_y'] = np.nanmedian(np.abs(v_gaze_y/v_tar_y))

                    # calculate the RMSE
                    _rx = ((_mix['x_estimated'] - _mix['smooth_pursuit_x']) / _mix['ppd_x']) ** 2
                    _ry = ((_mix['y_estimated'] - _mix['smooth_pursuit_y']) / _mix['ppd_y']) ** 2
                    _sp['rmse_x'] = np.sqrt(np.nanmean(_rx))
                    _sp['rmse_y'] = np.sqrt(np.nanmean(_ry))

                    # calculate the SNR
                    _snr_x, _snr_y = calculate_snr(_mix)
                    _sp['snr_x'] = _snr_x
                    _sp['snr_y'] = _snr_y

                    # save smooth pursuit features to file
                    _ev_sp_path = os.path.join(ev_folder, _task + '_' + _subj + '_extra' + '.csv')
                    _sp.to_csv(_ev_sp_path, index=False)


                    # # plot the gaze trace
                    # if _task=='sp':  # plot the x, y offset for the smooth pursuit task
                    #     _mix.plot(x='gt_ms', y=['x_deg', 'y_deg', 'tar_x', 'tar_y', 'offset_x', 'offset_y'])
                    #     _img_path = os.path.join(sp_folder, 'sp_trace_' + _subj + '.jpg')
                    #     plt.savefig(_img_path)
                    # else:
                    #     # show gaze traces with / without velocity traces
                    #     if show_vel:
                    #         _mix.plot(x='gt_ms', y=['x_deg', 'y_deg', 'x_vel', 'y_vel'])
                    #     else:
                    #         _mix.plot(x='gt_ms', y=['x_deg', 'y_deg'])

                    # plot eye events (saccades and fixations)
                    # if _task in ['fv', 'fs']: # skip fixation and saccade events plotting in the smooth pursuit task
                    #     for _idx, _row in _fix.iterrows():
                    #         x0 = _row['onset']
                    #         x1 = _row['offset']
                    #         plt.axvspan(x0, x1, color='blue', alpha=0.1, lw=0)
                    #
                    # plt.show()

                # plt.scatter(_sac['sac_amp'], _sac['sac_amp']/(_sac['duration']/1000))
                # plt.show()

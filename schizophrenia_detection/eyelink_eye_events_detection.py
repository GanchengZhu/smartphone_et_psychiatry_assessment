"""
eye-event detection with the I-DT algorithm, code adapted from 
https://github.com/aeye-lab/pymovements/blob/main/src/pymovements/events/detection/idt.py
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt


def butter_lowpass(data, cutoff_freq, sample_rate, order=4):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def calculate_snr(samples):
    """Calculate the SNR for the smooth pursuit task based on the
    samples - the sample files
    """

    sp_tar_x = samples['tx'].to_numpy()
    sp_tar_y = samples['ty'].to_numpy()
    sp_gaze_x = samples['gx'].to_numpy()
    sp_gaze_y = samples['gy'].to_numpy()

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
    SR = 1000

    # up-sample the data to 250 Hz
    _t = samples['ts_0ed'].to_numpy() / 1000
    tar_x = samples['tx'].to_numpy()
    tar_y = samples['ty'].to_numpy()
    gz_x = samples['gx'].to_numpy()
    gz_y = samples['gy'].to_numpy()

    # pixel per degree
    ppd_x = np.nanmean(samples['res_x'].to_numpy())
    ppd_y = np.nanmean(samples['res_y'].to_numpy())

    # estimate the gaze and target velocity in the x direction
    _vx = []
    for _x in [tar_x, gz_x]:
        _mask_x = ~np.isnan(_x)
        _time_x = _t[_mask_x]
        _clean_x = _x[_mask_x]
        cs_x = CubicSpline(_time_x, _clean_x)
        gz_x = cs_x(_t)

        # x direction
        n_x = len(gz_x)
        vel_x = np.full(n_x, 0.0)
        for i in range(4, n_x - 4):
            numerator = gz_x[i + 4] + gz_x[i + 3] + gz_x[i + 2] - gz_x[i - 2] - gz_x[i - 3] - gz_x[i - 4]
            denominator = 18 * ppd_x
            vel_x[i] = SR * (numerator / denominator)
            # print(vel_x[i])
        vel_x = butter_lowpass(vel_x, cutoff_freq=0.5, sample_rate=1000)
        _vx.append(vel_x)

    # estimate the gaze and target velocity in the y direction
    _vy = []
    for _y in [tar_y, gz_y]:
        _mask_y = ~np.isnan(_y)
        _time_y = _t[_mask_y]
        _clean_y = _y[_mask_y]
        cs_y = CubicSpline(_time_y, _clean_y)
        gz_y = cs_y(_t)

        # x direction
        n_y = len(gz_y)
        vel_y = np.full(n_y, 0.0)
        for i in range(4, n_y - 4):
            numerator = gz_y[i + 4] + gz_y[i + 3] + gz_y[i + 2] - gz_y[i - 2] - gz_y[i - 3] - gz_y[i - 4]
            denominator = 18 * ppd_y
            vel_y[i] = SR * (numerator / denominator)
        vel_y = butter_lowpass(vel_y, cutoff_freq=0.5, sample_rate=1000)
        _vy.append(vel_y)

    # plt.plot(_t,_vx[0])
    # plt.plot(_t,_vx[1])
    # print('velocity gain', np.nanmedian(np.abs(_vx[1]/_vx[0])))
    # plt.ylim([-20, 20])
    # plt.show()

    # return velocity_tar_x, velocity_gz_x, velocity_tar_y, velocity_gz_y
    return _vx + _vy


def get_events(asc_folder, event_folder):
    """
    Extract saccades and fixations from an EyeLink ASC file

    asc_folder -- The name of an ASC file
    event_folder -- the folder to save the event files
    """

    # get all files in the asc_folder into a glob
    glob_file_pattern = os.path.join('data_eyelink_asc', asc_folder, f"*.asc")
    files = glob.glob(glob_file_pattern)
    for file in files:
        # show the current process
        print("Processing...", file)

        _fn = os.path.basename(file)
        _fn = _fn.replace('1.asc', '.asc')
        _task, _subj, _ext = _fn.split('_')

        # open an ASC file and extract the fixations and saccades
        efix = []  # fixation end events
        esac = []  # saccade end events

        # Store samples in lists (timestamp, x, y, pupil size)
        smp = []

        _ts_start = []
        _ts_end = []
        with open(file) as asc:

            eye = 'R'  # default, record the right eye
            for line in asc:
                # check if we are recording the left eye instead
                if ('START' in line) and ('LEFT' in line):
                    eye = 'L'

                # Extract all numbers and put them in a list
                tmp_data = [float(x) for x in re.findall(r'-?\d+\.?\d*', line)]

                # look for trial start/end messages, extract the timestamps
                if _task == 'fx':
                    if 'fixcross_onset' in line:
                        _ts_start.append(tmp_data[0])
                if _task == 'fv':
                    if 'image_onset' in line:
                        _ts_start.append(tmp_data[0])
                if _task == 'ps':
                    if 'movement_onset' in line:
                        _ts_start.append(tmp_data[0])

                if 'blank_screen' in line:
                    _ts_end.append(tmp_data[0])

                # retrieve events parsed from the right eye recording
                if re.search(f'^EFIX {eye}', line):
                    # handling invalid fixations
                    if len(tmp_data) < 8:
                        tmp_data = ['NaN'] * 8
                    else:
                        efix.append(tmp_data)

                elif re.search(f'^ESACC {eye}', line):
                    # handling invalid saccades
                    if len(tmp_data) < 11:
                        tmp_data = ['NaN'] * 11
                    else:
                        esac.append(tmp_data)
                else:
                    pass

                # get the samples
                if re.search('^\d', line) and (_task == 'ps'):
                    # 1241262	  956.8	  289.2	  600.0	    3.3	   -0.4	  46.60	  46.50	...
                    # 1241402	  957.1	  290.4	  640.0	   .	   .	  46.60	  46.50	...
                    if len(tmp_data) == 13:
                        smp.append([tmp_data[i] for i in [0, 4, 5, 6, 9, 10, 11, 12]])
                        # print(tmp_data, len(tmp_data))
                    elif len(tmp_data) == 8:  # normal sample line
                        smp.append(tmp_data)
                        # print(tmp_data, len(tmp_data))
                    else:  # sample line with missing values (e.g., tracking loss)
                        smp.append([tmp_data[0]] + [np.nan] * 7)

        # handle exceptions, where the start and end messages do not match in the EDF data file
        if ('fv_245_1.asc' in file) or ('ps_254_1.asc' in file):
            _ts_end.pop(0)

        # print(file,  len(_ts_start)== len(_ts_end), len(_ts_start), len(_ts_end))
        # if not (len(_ts_start)== len(_ts_end)):
        #     print(_ts_start)
        #     print(_ts_end)

        # Put the extracted data into pandas data frames
        # EFIX R 80790054 80790349 296 981.3 554.5 936
        # onset,offset,start_x,start_y,end_x,end_y,avg_x,avg_y,duration,dispersion,spl,avg_vel
        efix_colname = ['onset', 'offset', 'duration', 'avg_x', 'avg_y', 'avgPupil', 'res_x', 'res_y']
        efixFRM = pd.DataFrame(efix, columns=efix_colname)

        # ESACC R 80790350 80790372 23 982.6 551.8 864.9 587.9 1.94 151
        # onset,offset,start_x,start_y,end_x,end_y,sac_amp,duration,peakv,dispersion,spl,avg_vel
        esac_colname = ['onset', 'offset', 'duration', 'start_x', 'start_y',
                        'end_x', 'end_y', 'sac_amp', 'peakv', 'res_x', 'res_y']
        esacFRM = pd.DataFrame(esac, columns=esac_colname)

        # exclude fixations and saccades outside the trial/image time windows
        _fix = pd.DataFrame(columns=efixFRM.columns)
        _sac = pd.DataFrame(columns=esacFRM.columns)

        for i in range(len(_ts_start)):
            _f = efixFRM[(efixFRM['offset'] >= _ts_start[i]) & (efixFRM['onset'] <= _ts_end[i])]
            _e = esacFRM[(esacFRM['offset'] >= _ts_start[i]) & (esacFRM['onset'] <= _ts_end[i])]
            if len(_fix) == 0:
                _fix = _f
            else:
                if len(_fix) == 0:
                    pass
                else:
                    _fix = pd.concat([_fix, _f], ignore_index=True)

            if len(_sac) == 0:
                _sac = _e
            else:
                if len(_sac) == 0:
                    pass
                else:
                    _sac = pd.concat([_sac, _e], ignore_index=True)

        # save data to .csv files in events/data_eyelink
        _fix_path = os.path.join(event_folder, f'fix_{_subj}_{_task}.csv')
        _sac_path = os.path.join(event_folder, f'sac_{_subj}_{_task}.csv')
        _fix.to_csv(_fix_path)
        _sac.to_csv(_sac_path)

        if _task == 'ps':
            # set the parameter of the curve
            amp_x, amp_y = (1920 * 0.246 / 2, 1080 * 0.881 / 2)  # match the phone screen in dva
            freq_x, freq_y = (1 / 8.0, 1 / 12.0)
            phi = 0
            phase_x, phase_y = (np.pi * phi, 0)

            # let's deal with the samples
            smp_colnames = ['ts', 'gx', 'gy', 'pupil', 'vx', 'xy', 'res_x', 'res_y']
            smpFRM = pd.DataFrame(smp, columns=smp_colnames)

            if len(smpFRM) == 0:
                continue

            smpFRM = smpFRM[(smpFRM['ts'] >= _ts_start[0]) & (smpFRM['ts'] <= _ts_end[0])]
            smpFRM['ts_0ed'] = smpFRM['ts'] - smpFRM.iloc[0]['ts']
            smpFRM['tx'] = 1920 / 2.0 + amp_x * np.sin(2 * np.pi * freq_x * smpFRM['ts_0ed'] / 1000. + phase_x)
            smpFRM['ty'] = 1080 / 2.0 + amp_y * np.sin(2 * np.pi * freq_y * smpFRM['ts_0ed'] / 1000. + phase_y)

            # smpFRM.plot('ts_0ed', ['gx', 'gy', 'tx', 'ty'])
            # plt.show()

            # calculate the x, y offsets
            smpFRM['offset_x'] = np.abs(smpFRM['gx'] - smpFRM['tx']) / smpFRM['res_x']
            smpFRM['offset_y'] = np.abs(smpFRM['gy'] - smpFRM['ty']) / smpFRM['res_y']
            _sp = pd.DataFrame({'offset_x_mean': [np.mean(smpFRM['offset_x'])],
                                'offset_x_p80': [smpFRM['offset_x'].quantile(0.5)],
                                'offset_x_std': [np.std(smpFRM['offset_x'])],
                                'offset_y_mean': [np.mean(smpFRM['offset_y'])],
                                'offset_y_p80': [smpFRM['offset_y'].quantile(0.5)],
                                'offset_y_std': [np.std(smpFRM['offset_y'])],
                                'cor_x': smpFRM.corr()['tx']['gx'],
                                'cor_y': smpFRM.corr()['ty']['gy']})

            # calculate velocity gain
            v_tar_x, v_gaze_x, v_tar_y, v_gaze_y = calculate_instantaneous_velocity(smpFRM)
            # exclude the beginning and ending 0.5 sec. to reduce noise due to up-sampling and filtering
            v_tar_x = v_tar_x[125:-125]
            v_gaze_x = v_gaze_x[125:-125]
            v_tar_y = v_tar_y[125:-125]
            v_gaze_y = v_gaze_y[125:-125]

            v_tar_x = np.where(v_tar_x == 0, np.nan, v_tar_x).astype(float)
            v_tar_y = np.where(v_tar_y == 0, np.nan, v_tar_y).astype(float)
            # v_tar_x[v_tar_x == 0] = np.nan
            # v_tar_y[v_tar_y == 0] = np.nan
            # print('velocity', len(v_tar_x), len(v_gaze_x))

            _sp['gain_x'] = np.nanmedian(np.abs(v_gaze_x / v_tar_x))
            _sp['gain_y'] = np.nanmedian(np.abs(v_gaze_y / v_tar_y))

            # calculate the RMSE
            _rx = ((smpFRM['gx'] - smpFRM['tx']) / smpFRM['res_x']) ** 2
            _ry = ((smpFRM['gy'] - smpFRM['ty']) / smpFRM['res_y']) ** 2
            _sp['rmse_x'] = np.sqrt(np.nanmean(_rx))
            _sp['rmse_y'] = np.sqrt(np.nanmean(_ry))

            # calculate the SNR
            _snr_x, _snr_y = calculate_snr(smpFRM)
            _sp['snr_x'] = _snr_x
            _sp['snr_y'] = _snr_y

            # save smooth pursuit features to file
            _sp_path = os.path.join(ev_folder, f'sp_{_subj}_extra.csv')
            _sp.to_csv(_sp_path, index=False)

            # print(smpFRM)

        # smpFRM.plot('ts', ['gx', 'gy'])
        # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', help='raw data path')
    args = parser.parse_args()

    # base folder
    base_path = os.path.join(os.getcwd(), 'data_eyelink_asc', args.raw_data_path)

    # create a folder to save the detected events
    ev_folder = os.path.join(os.getcwd(), 'events', 'data_eyelink', args.raw_data_path)
    os.makedirs(ev_folder, exist_ok=True)

    get_events(base_path, ev_folder)

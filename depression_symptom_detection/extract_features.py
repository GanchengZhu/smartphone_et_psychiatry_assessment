"""
Extract typically used eye event features
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

# testing the algorithm with an input file

if __name__ == "__main__":
    # image list
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

    _im_num = []
    _im_pat = []
    for _im in image_list:
        _im = _im.replace('depression_images/', '')
        _im = _im.strip('.jpg')
        _tb, _num = _im.split('_')
        # print(_tb, _num)
        _im_num.append(int(_num))
        _im_pat.append(_tb)

    cols = ['subj']
    for i in range(24):
        cols.extend([
            f'Trl_{i}_Fix_Count_L',
            f'Trl_{i}_Fix_Duration_Mean_L',
            f'Trl_{i}_Fix_Duration_Std_L',
            f'Trl_{i}_Fix_X_Mean_L',
            f'Trl_{i}_Fix_X_Std_L',
            f'Trl_{i}_Fix_Y_Mean_L',
            f'Trl_{i}_Fix_Y_Std_L',
            f'Trl_{i}_Fix_Count_H',
            f'Trl_{i}_Fix_Duration_Mean_H',
            f'Trl_{i}_Fix_Duration_Std_H',
            f'Trl_{i}_Fix_X_Mean_H',
            f'Trl_{i}_Fix_X_Std_H',
            f'Trl_{i}_Fix_Y_Mean_H',
            f'Trl_{i}_Fix_Y_Std_H',
        ])

    cols.extend([
        'Fix_Count',
        'Fix_Duration_Mean',
        'Fix_Duration_Std',
        'Fix_X_Mean',
        'Fix_X_Std',
        'Fix_Y_Mean',
        'Fix_Y_Std',
    ])
    fix_fv = pd.DataFrame(columns=cols)

    # data frame to store saccade data
    sac_fv = pd.DataFrame(columns=['subj',
                                   'Sac_Duration_Mean',
                                   'Sac_Duration_Std',
                                   'Sac_Amplitude_Mean',
                                   'Sac_Amplitude_Std',
                                   'Sac_Velocity_Mean',
                                   'Sac_Velocity_Std'
                                   ])

    # folder that stores the detected eye event files
    ev_folder = os.path.join(os.getcwd(), 'events')

    _ev_files = os.listdir(ev_folder)
    for _file in _ev_files:
        if _file == ".DS_Store":
            continue

        # get task, subj #, and data type from the file name
        _dt, _subj, _task = _file.split('.')[0].split('_')

        # full path to the data file
        _path = os.path.join(ev_folder, _file)

        # processing fixation data 
        if _dt == 'fix':
            # read in the fixation data file
            _fix = pd.read_csv(_path)
            _fix_cnt = len(_fix['duration'])
            _fix_duration_mean = np.nanmean(_fix['duration'])
            _fix_duration_std = np.nanstd(_fix['duration'])
            _fix_avg_x_mean = np.nanmean(_fix['avg_x'])
            _fix_avg_x_std = np.nanstd(_fix['avg_x'])
            _fix_avg_y_mean = np.nanmean(_fix['avg_y'])
            _fix_avg_y_std = np.nanstd(_fix['avg_y'])
            _fix_dispersion = np.nanmean(pdist(_fix[['avg_x', 'avg_y']], metric='euclidean'))

            if _fix_cnt == 0:
                _fix_avg_x_mean = 0.0
                _fix_avg_x_std = 0.0
                _fix_avg_y_mean = 0.0
                _fix_avg_y_std = 0.0
                _fix_dispersion = 0.0

            if _task == 'FVTask':  # record data for the smooth pursuit task
                _features = [_subj]
                for i in range(24):
                    _trial_start = i * 5000 + 1000
                    _trial_end = (i + 1) * 5000
                    # extract rows that belong to a single trial
                    _td = _fix.loc[(_fix['offset'] >= _trial_start) & (_fix['offset'] <= _trial_end)]
                    # # extract rows with fixation in the top AOI
                    # _top = _td.loc[_td['avg_y'] < 12.4]
                    # _bot = _td.loc[_td['avg_y'] >= 12.4]

                    if _im_pat[i] == 'HL':
                        _H = _td.loc[_td['avg_y'] < 11.5]
                        _L = _td.loc[_td['avg_y'] >= 11.5]
                    else:
                        _L = _td.loc[_td['avg_y'] < 11.5]
                        _H = _td.loc[_td['avg_y'] >= 11.5]

                    _tmp = [len(_L['duration']),
                            np.mean(_L['duration']),
                            np.std(_L['duration']),
                            np.mean(_L['avg_x']),
                            np.std(_L['avg_x']),
                            np.mean(_L['avg_y']),
                            np.std(_L['avg_y']),
                            len(_H['duration']),
                            np.mean(_H['duration']),
                            np.std(_H['duration']),
                            np.mean(_H['avg_x']),
                            np.std(_H['avg_x']),
                            np.mean(_H['avg_y']),
                            np.std(_H['avg_y'])]

                    if _tmp[0] == 0:
                        _tmp[1] = 0
                        _tmp[2] = 0
                        _tmp[3] = 0
                        _tmp[4] = 0
                        _tmp[5] = 0
                        _tmp[6] = 0
                    if _tmp[7] == 0:
                        _tmp[8] = 0
                        _tmp[9] = 0
                        _tmp[10] = 0
                        _tmp[11] = 0
                        _tmp[12] = 0
                        _tmp[13] = 0

                    _features.extend(_tmp)

                _exp_features = [_fix_cnt,
                                 _fix_duration_mean,
                                 _fix_duration_std,
                                 _fix_avg_x_mean,
                                 _fix_avg_x_std,
                                 _fix_avg_y_mean,
                                 _fix_avg_y_std,
                                 # _fix_dispersion,
                                 ]
                _features.extend(_exp_features)

                fix_fv.loc[len(fix_fv)] = _features

        # processing saccade data
        if _dt == 'sac':
            # read in the fixation data file
            _sac = pd.read_csv(_path)
            _sac_cnt = len(_sac['duration'])
            _sac_amp_mean = np.nanmean(_sac['sac_amp'])
            _sac_amp_std = np.nanstd(_sac['sac_amp'])

            _sac_duration_mean = np.nanmean(_sac['duration'])
            _sac_duration_std = np.nanstd(_sac['duration'])
            _sac['vel'] = _sac['sac_amp'] / _sac['duration']  # velocity
            _sac_vel_mean = np.nanmean(_sac['vel']) * 1000
            _sac_vel_std = np.nanstd(_sac['vel'])

            if _task == 'FVTask':
                sac_fv.loc[len(sac_fv)] = [_subj,
                                           # _sac_cnt,
                                           _sac_duration_mean,
                                           _sac_duration_std,
                                           _sac_amp_mean,
                                           _sac_amp_std,
                                           _sac_vel_mean,
                                           _sac_vel_std]

        # # save results to csv
    feature_dir = os.path.join(os.path.dirname(__file__), 'features')
    os.makedirs(feature_dir, exist_ok=True)
    _fv_fix_path = os.path.join(feature_dir, 'fv_fixation_features.csv')
    _fv_sac_path = os.path.join(feature_dir, 'fv_saccade_features.csv')
    fix_fv.to_csv(_fv_fix_path, index=False)
    sac_fv.to_csv(_fv_sac_path, index=False)

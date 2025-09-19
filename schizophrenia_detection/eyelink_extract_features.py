import argparse
import glob
import os
from functools import reduce

import numpy as np
import pandas as pd
from scipy.stats import chi2

# FIX_FEATURES = ["fix_dur", "fix_spl", "fix_dispersion", "fix_av"]
# SAC_FEATURES = ["sac_dur", "sac_amp", "sac_pkv", "sac_dispersion", "sac_av", "sac_spl"]
SP_FEATURES = ['offset_x_mean', 'offset_x_std', 'offset_y_mean', 'offset_y_std',
               'cor_x', 'cor_y', 'gain_x', 'gain_y', 'rmse_x', 'rmse_y', 'snr_x', 'snr_y']

STATS = ['mean', 'std', 'median', 'q75', 'q25']
# STATS = ['mean', 'std', 'q95', 'q5', 'median', 'q75', 'q25', 'max', 'min']
# STATS = ['mean', 'std', 'q95', 'q5', 'median', 'q75', 'q25', ]
# STATS = ['mean', 'std', 'median', 'q75', 'q25']

# loop over the 3 three task and store the fixation and saccade data into csv files
_FILES = [('fx', 'fix'),
          ('fx', 'sac'),
          ('fv', 'fix'),
          ('fv', 'sac'),
          ('ps', 'fix'),
          ('ps', 'sac'),
          ('extra', 'sp')]


def extract_stats(arr: np.ndarray) -> list:
    """
    Extract 7 statistics along the row dimension.
    """

    if isinstance(arr, pd.DataFrame):
        arr = np.array(arr)
    if len(arr) == 0:
        stats = [np.nan] * 9
    else:
        stats = [
            np.nanmean(arr, axis=0),
            np.nanstd(arr, axis=0),
            # np.nanmax(arr, axis=0),
            # np.nanmin(arr, axis=0),
            # np.nanquantile(arr, 0.95, axis=0),
            # np.nanquantile(arr, 0.05, axis=0),
            np.nanmedian(arr, axis=0),
            np.nanquantile(arr, 0.75, axis=0),
            np.nanquantile(arr, 0.25, axis=0),
        ]
    # res = stats.tolist()
    return stats


def process_all_events(ev_folder: str) -> pd.DataFrame:
    """
    Process all fix/sac event files in the folder and return a combined feature DataFrame.
    """

    """ combine all data files of the same type and save into a single csv file"""
    # ev_folder could be one of the following
    # batch_0 = os.path.join(os.getcwd(), 'events', 'data_phone', 'batch_0')
    # batch_1 = os.path.join(os.getcwd(), 'events', 'data_phone', 'batch_1')
    # batch_r = os.path.join(os.getcwd(), 'events', 'data_phone', 'batch_1_sz_repeat_measure')

    _data_frames = []
    for task, data in _FILES:
        file_pattern = os.path.join(ev_folder, f"{data}_*_{task}.csv")  # Adjust pattern (e.g., "folder/data_*.csv")
        file_paths = glob.glob(file_pattern)

        # store the subject-level data in a matrix
        _subj_data = []
        for file in file_paths:
            print(file)
            # extract the subject id
            parts = file.split('.')[0].split('_')  # e.g., fix_001_fs
            subj_id = parts[-2]

            df = pd.read_csv(file)  # Use pd.read_excel() for Excel files
            # event file header
            # FIX: onset,offset,start_x,start_y,end_x,end_y,avg_x,avg_y,duration,dispersion,spl,avg_vel
            # SAC: sac_amp,duration,peakv,dispersion,spl,avg_vel
            # SP_ETRA: offset_x_mean, offset_x_std,offset_y_mean,offset_y_std,
            #          cor_x,cor_y,gain_x,gain_y,rmse_x,rmse_y,snr_x,snr_y

            if data == 'fix':
                _dur_stats = extract_stats(df['duration'])
                _dwell = np.nansum(df['duration'])

                # quantify dispersion with BCAE
                df['avg_x'] = df['avg_x'] / df['res_x']  # convert to degree
                df['avg_x'] = df['avg_x'] / df['res_x']  # convert to degree

                sigma_x = np.std(df['avg_x'])
                sigma_y = np.std(df['avg_y'])
                rho = df[['avg_x', 'avg_y']].corr()  # Automatically ignores NaN pairs
                rho = rho.loc['avg_x', 'avg_y']
                chi_square = chi2.ppf(0.682, 2)
                _dispersion = np.pi * chi_square * sigma_x * sigma_y * np.sqrt(1 - rho ** 2)  # BCAE
                # add the subject-level data to a matrix
                _subj_data.append([subj_id] + [len(df['duration'])] + _dur_stats + [_dwell, _dispersion])

            if data == 'sac':
                _dur_stats = extract_stats(df['duration'])
                _amp_stats = extract_stats(df['sac_amp'])
                _peakv_stats = extract_stats(df['peakv'])
                _spl = np.nansum(df['sac_amp'])
                _subj_data.append([subj_id] + _dur_stats + _amp_stats + _peakv_stats + [_spl])

            if data == 'sp':
                df = df[SP_FEATURES]
                _sp_metrics = df.iloc[0].to_list()
                _subj_data.append([subj_id] + _sp_metrics)

        if task == 'ps':
            task = 'sp'
        elif task == 'fx':
            task = 'fs'

        if data == 'fix':
            col = ['subj_id', f'{task}_fix_cnt'] + [f'{task}_fix_dur_{stats}' for stats in STATS] + \
                  [f'{task}_fix_dwell_time', f'{task}_fix_dispersion']
        if data == 'sac':
            col = ['subj_id'] + [f'{task}_sac_dur_{stats}' for stats in STATS] + \
                  [f'{task}_sac_amp_{stats}' for stats in STATS] + \
                  [f'{task}_sac_peakv_{stats}' for stats in STATS] + \
                  [f'{task}_sac_SPL']
        if data == 'sp':
            col = ['subj_id'] + [f'sp_{stats}' for stats in SP_FEATURES]

        dataframe = pd.DataFrame(_subj_data, columns=col)
        # dataframe.to_csv(f'_tmp_{task}_{data}.csv')
        _data_frames.append(dataframe)

    merged_df = reduce(lambda left, right: pd.merge(left, right, on='subj_id'), _data_frames)

    return merged_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', help='Raw data folder under events/data_phone/')
    args = parser.parse_args()

    # 输入路径：events/data_phone/xxx/
    ev_folder = os.path.join('events', 'data_eyelink', args.raw_data_path)
    # assert os.path.exists(ev_folder), f"Path does not exist: {ev_folder}"

    feature_df = process_all_events(ev_folder)

    # 输出路径：features/data_phone/xxx.xlsx
    os.makedirs(os.path.join('features', 'data_eyelink'), exist_ok=True)
    out_path = os.path.join('features', 'data_eyelink', f"{args.raw_data_path}.xlsx")
    feature_df.to_excel(out_path, index=False)

    print(f"Saved features to {out_path}")

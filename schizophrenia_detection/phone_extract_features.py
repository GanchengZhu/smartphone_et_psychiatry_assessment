import argparse
import glob
import os
from functools import reduce

import numpy as np
import pandas as pd
from scipy.stats import chi2

FIX_FEATURES = ["fix_dur", "fix_spl", "fix_dispersion", "fix_av"]
SAC_FEATURES = ["sac_dur", "sac_amp", "sac_pkv", "sac_dispersion", "sac_av", "sac_spl"]
SP_FEATURES = ['offset_x_mean', 'offset_x_std', 'offset_y_mean', 'offset_y_std',
               'cor_x', 'cor_y', 'gain_x', 'gain_y', 'rmse_x', 'rmse_y', 'snr_x', 'snr_y']

STATS = ['mean', 'std', 'median', 'q75', 'q25']
# STATS = ['mean', 'std', 'max', 'min', 'q95', 'q5', 'median', 'q75', 'q25']
# STATS = ['mean', 'std', 'median', 'q75', 'q25']

# loop over the 3 three task and store the fixation and saccade data into csv files
_FILES = [('fs', 'fix'),
          ('fs', 'sac'),
          ('fv', 'fix'),
          ('fv', 'sac'),
          ('sp', 'fix'),
          ('sp', 'sac'),
          ('extra', 'sp')]


def extract_stats(arr: np.ndarray) -> list:
    """
    Extract 7 statistics along the row dimension.
    """

    if isinstance(arr, pd.DataFrame):
        arr = np.array(arr)

    stats = [
        np.nanmean(arr, axis=0),
        np.nanstd(arr, axis=0),
        # np.nanmax(arr, axis=0),
        # np.nanmin(arr, axis=0),
        # np.nanquantile(arr, 0.95, axis=0),
        # np.nanquantile(arr, 0.05, axis=0),
        np.nanmedian(arr, axis=0),
        np.nanquantile(arr, 0.75, axis=0),
        np.nanquantile(arr, 0.25, axis=0)
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
            # SAC: onset,offset, sac_amp,duration,peakv,dispersion,spl,avg_vel
            # SP_ETRA: offset_x_mean, offset_x_std,offset_y_mean,offset_y_std,
            #          cor_x,cor_y,gain_x,gain_y,rmse_x,rmse_y,snr_x,snr_y

            if '_fv' in file:
                # 初始化为空，但保持 df 的列和类型
                processed_df = None

                for trial_id in range(20):
                    trial_start = trial_id * 6000 + 1000  # 跳过前1000ms
                    trial_end = (trial_id + 1) * 6000

                    # 找出与当前 trial 有重叠的事件
                    overlapping = df[(df['offset'] > trial_start) & (df['onset'] < trial_end)].copy()

                    if overlapping.empty:
                        continue  # 没有数据就跳过当前 trial

                    # 修正跨越 trial 边界的事件
                    overlapping['onset'] = overlapping['onset'].clip(lower=trial_start)
                    overlapping['offset'] = overlapping['offset'].clip(upper=trial_end)

                    # 添加 trial_id（可选）
                    overlapping['trial_id'] = trial_id

                    # 拼接到总表
                    if processed_df is None:
                        processed_df = overlapping
                    else:
                        processed_df = pd.concat([processed_df, overlapping], ignore_index=True)

                if processed_df is not None:
                    df = processed_df.reset_index(drop=True)
                else:
                    df = pd.DataFrame(columns=df.columns)  # 如果最终是空的，返回空表

            if data == 'fix':
                _dur_stats = extract_stats(df['duration'])
                _dwell = np.nansum(df['duration'])
                # quantify dispersion with BCAE
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

    # # merge dataframes
    # merged_df = pd.concat(dataframes, ignore_index=True)
    # # save merged data frames
    # merged_df.to_csv(f"{data}_{task}_all.csv", index=False)
    #
    #
    # # dict[subject_id] = dict[column_name] = value
    # subject_data = defaultdict(dict)
    # subj_ids = set()
    # print(ev_folder)
    # for filename in os.listdir(ev_folder):
    #     if not filename.endswith('.csv') or filename.startswith('.'):
    #         continue
    #
    #     parts = filename.split('.')[0].split('_')  # e.g., fix_001_fs
    #     if len(parts) != 3:
    #         continue
    #
    #     ev_type, subj_id, task = parts
    #     subj_ids.add(subj_id)
    #
    #     filepath = os.path.join(ev_folder, filename)
    #     df = pd.read_csv(filepath)
    #
    #     # print(filename)
    #     if 'extra' in filename:
    #         for feature_name in ['cor_x', 'cor_y', 'gain_x', 'gain_y', 'rmse_x', 'rmse_y', 'snr_x', 'snr_y']:
    #             subject_data[subj_id]['sp_' + feature_name] = df[feature_name].values[0]
    #     else:
    #         # if ev_type == 'sac':
    #         try:
    #             df = df[df['spl'] != 0]
    #             df = df[df['dispersion'] != 0]
    #         except:
    #             pass
    #         # if ev_type == 'sac':
    #         #     df = df[df['duration'] ]
    #
    #         # Extract relevant feature columns
    #         if ev_type == 'fix':
    #             features = [
    #                 df['duration'].values,
    #                 df['spl'].values,
    #                 df['dispersion'].values,
    #                 df['avg_vel'].values
    #             ]
    #             prefix = 'fix'
    #             feature_names = FIX_FEATURES
    #             count_col = f"{task}_{prefix}_count"
    #         elif ev_type == 'sac':
    #             features = [
    #                 df['duration'].values,
    #                 df['sac_amp'].values,
    #                 df['peakv'].values,
    #                 df['dispersion'].values,
    #                 df['avg_vel'].values,
    #                 df['spl'].values
    #             ]
    #             prefix = 'sac'
    #             feature_names = SAC_FEATURES
    #             count_col = None  # saccade 不需要 count 列
    #         else:
    #             continue
    #
    #         data = np.stack(features, axis=1)
    #         stats = extract_stats(data)
    #
    #         # 每个列的名字
    #         col_names = [f"{task}_{feat}_{stat}" for stat in STATS for feat in feature_names]
    #         # print(col_names)
    #         if prefix == 'fix':
    #             subject_data[subj_id][count_col] = data.shape[0]
    #         for name, value in zip(col_names, stats):
    #             subject_data[subj_id][name] = value
    #
    # # 构建 DataFrame
    # all_records = []
    # for subj_id in sorted(subj_ids):
    #     row = {'subj_id': subj_id}
    #     row.update(subject_data[subj_id])
    #     all_records.append(row)
    #
    # return pd.DataFrame(all_records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_path', help='Raw data folder under events/data_phone/')
    args = parser.parse_args()

    # 输入路径：events/data_phone/xxx/
    ev_folder = os.path.join('events', 'data_phone', args.raw_data_path)
    # assert os.path.exists(ev_folder), f"Path does not exist: {ev_folder}"

    feature_df = process_all_events(ev_folder)

    # 输出路径：features/data_phone/xxx.xlsx
    os.makedirs(os.path.join('features', 'data_phone'), exist_ok=True)
    out_path = os.path.join('features', 'data_phone', f"{args.raw_data_path}.xlsx")
    feature_df.to_excel(out_path, index=False)

    print(f"Saved features to {out_path}")

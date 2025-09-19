import os
from math import sqrt

import pandas as pd
import pingouin as pg  # 推荐安装: pip install pingouin
import scipy.stats as stats

save_dir = 'sac_main_seq'
# 载入数据
df = pd.read_csv(os.path.join(save_dir, 'saccade_gof_summary.csv'))


# ----------- Part 1: 差异分析（schizophrenia vs control） -----------
def compute_effect_size_ttest(group1, group2):
    n1, n2 = len(group1), len(group2)
    pooled_std = sqrt(((n1 - 1) * group1.std() ** 2 + (n2 - 1) * group2.std() ** 2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std


def analyze_r2_by_group(metric, device):
    sub = df[df['device'] == device]
    sz = sub[sub['label'] == 1][f'{metric}_r2'].dropna()
    hc = sub[sub['label'] == 0][f'{metric}_r2'].dropna()

    # 正态性检测
    p1 = stats.shapiro(sz)[1] if len(sz) > 3 else 1
    p2 = stats.shapiro(hc)[1] if len(hc) > 3 else 1
    normal = (p1 > 0.05 and p2 > 0.05)

    t, p = stats.ttest_ind(sz, hc, equal_var=False)
    d = pg.compute_effsize(sz, hc, paired=False, eftype='cohen')
    print(f"{device} {metric}: t-test p={p:.4f}, Cohen's d={d:.3f}")

    print(f"\n--- {device.upper()} {metric.upper()} R² (SZ vs HC) ---")
    print(f"SZ (n={len(sz)}): Mean = {sz.mean():.3f}, SD = {sz.std():.3f}")
    print(f"HC (n={len(hc)}): Mean = {hc.mean():.3f}, SD = {hc.std():.3f}")
    # if normal:
    #     t, p = stats.ttest_ind(sz, hc, equal_var=False)
    #     d = pg.compute_effsize(sz, hc, paired=False, eftype='cohen')
    #     print(f"{device} {metric}: t-test p={p:.4f}, Cohen's d={d:.3f}")
    # else:
    #     u, p = stats.mannwhitneyu(sz, hc, alternative='two-sided')
    #     d = pg.compute_effsize(sz, hc, paired=False, eftype='cles')
    #     print(f"{device} {metric}: Mann‑Whitney U p={p:.4f}, CLES={d:.3f}")


# for metric in ['dur', 'vel']:
#     for device in ['eyelink', 'phone']:
#         analyze_r2_by_group(metric, device)


# ----------- Part 2: phone vs eyelink 的相关性分析 -----------
# def analyze_correlation(metric: str):
#     # 只保留两个设备都可用的数据
#     df_both = df[df['phone_both'] == 1]
#     df_both = df_both[df_both['eyelink_both'] == 1]
#
#     phone_df = df_both[df_both['device'] == 'phone'][['id', f'{metric}_r2']].rename(
#         columns={f'{metric}_r2': 'r2_phone'})
#     eye_df = df_both[df_both['device'] == 'eyelink'][['id', f'{metric}_r2']].rename(
#         columns={f'{metric}_r2': 'r2_eyelink'})
#
#     merged = pd.merge(phone_df, eye_df, on='id').dropna()
#
#     print(f"\n--- CORRELATION of {metric.upper()} R² between phone and eyelink ---")
#     if len(merged) < 3:
#         print("Not enough paired data.")
#         return
#
#     # 使用皮尔逊和斯皮尔曼相关性
#     pearson_r, pearson_p = stats.pearsonr(merged['r2_phone'], merged['r2_eyelink'])
#     spearman_r, spearman_p = stats.spearmanr(merged['r2_phone'], merged['r2_eyelink'])
#
#     t, p = stats.ttest_ind(merged['r2_phone'], merged['r2_eyelink'], equal_var=False)
#     d = pg.compute_effsize(merged['r2_phone'], merged['r2_eyelink'], paired=False, eftype='cohen')
#
#     print(f"{metric} difference: t is ", t, 'p is ', p)
#
#     print(f"{device} {metric}: t-test p={p:.4f}, Cohen's d={d:.3f}")
#     print(f"Pearson: r = {pearson_r:.3f}, p = {pearson_p:.4f}")
#     print(f"Spearman: r = {spearman_r:.3f}, p = {spearman_p:.4f}")
#
#     # 提取数据
#     phone_r2 = df_both[df_both['device'] == 'phone'][f'{metric}_r2']
#     eye_r2 = df_both[df_both['device'] == 'eyelink'][f'{metric}_r2']
#     print(f"\n--- CORRELATION of {metric.upper()} R² (Phone vs Eyelink) ---")
#     print(f"Phone (n={len(phone_r2)}): Mean R² = {phone_r2.mean():.3f}, SD = {phone_r2.std():.3f}")
#     print(f"Eyelink (n={len(eye_r2)}): Mean R² = {eye_r2.mean():.3f}, SD = {eye_r2.std():.3f}")


# for metric in ['dur', 'vel']:
#     for para in [0, 1]:
#         analyze_correlation(metric)


def analyze_para(metric: str, para_id):
    df_both = df[df['phone_both'] == 1]
    df_both = df_both[df_both['eyelink_both'] == 1]

    phone_df = df_both[df_both['device'] == 'phone'][['id', f'{metric}_para_{para_id}']].rename(
        columns={f'{metric}_para_{para_id}': 'para_phone'})
    eye_df = df_both[df_both['device'] == 'eyelink'][['id', f'{metric}_para_{para_id}']].rename(
        columns={f'{metric}_para_{para_id}': 'para_eyelink'})

    merged = pd.merge(phone_df, eye_df, on='id').dropna()

    print(f"\n--- CORRELATION of {metric.upper()} PARA {para_id} between phone and eyelink ---")
    if len(merged) < 3:
        print("Not enough paired data.")
        return

    # 使用皮尔逊和斯皮尔曼相关性
    pearson_r, pearson_p = stats.pearsonr(merged['para_phone'], merged['para_eyelink'])
    spearman_r, spearman_p = stats.spearmanr(merged['para_phone'], merged['para_eyelink'])

    t, p = stats.ttest_ind(merged['para_phone'], merged['para_eyelink'], equal_var=False)
    d = pg.compute_effsize(merged['para_phone'], merged['para_eyelink'], paired=False, eftype='cohen')

    print(f"{metric} para {para_id} difference: t is ", t, 'p is ', p)
    print(f"{metric} para {para_id} : t-test p={p:.4f}, Cohen's d={d:.3f}")
    print(f"Pearson: r = {pearson_r:.3f}, p = {pearson_p:.4f}")
    print(f"Spearman: r = {spearman_r:.3f}, p = {spearman_p:.4f}")

    # 提取数据
    phone_para = df_both[df_both['device'] == 'phone'][f'{metric}_para_{para_id}']
    eye_para = df_both[df_both['device'] == 'eyelink'][f'{metric}_para_{para_id}']
    print(f"\n--- CORRELATION of {metric.upper()} PARA {para_id} (Phone vs Eyelink) ---")
    print(f"Phone (n={len(phone_para)}): Mean PARA {para_id} = {phone_para.mean():.3f}, SD = {phone_para.std():.3f}")
    print(f"Eyelink (n={len(eye_para)}): Mean PARA {para_id} = {eye_para.mean():.3f}, SD = {eye_para.std():.3f}")


def analyze_group_diff_by_para(metric, device, para_id):
    sub = df[df['device'] == device]
    sz = sub[sub['label'] == 1][f'{metric}_para_{para_id}'].dropna()
    hc = sub[sub['label'] == 0][f'{metric}_para_{para_id}'].dropna()

    # 正态性检测
    p1 = stats.shapiro(sz)[1] if len(sz) > 3 else 1
    p2 = stats.shapiro(hc)[1] if len(hc) > 3 else 1
    normal = (p1 > 0.05 and p2 > 0.05)

    # t, p = stats.ttest_ind(sz, hc, equal_var=False)
    # d = pg.compute_effsize(sz, hc, paired=False, eftype='cohen')
    # print(f"{device} {metric}: t-test p={p:.4f}, Cohen's d={d:.3f}")

    print(f"\n--- {device.upper()} {metric.upper()} PARA {para_id} (SZ vs HC) ---")
    print(f"SZ (n={len(sz)}): Mean = {sz.mean():.3f}, SD = {sz.std():.3f}")
    print(f"HC (n={len(hc)}): Mean = {hc.mean():.3f}, SD = {hc.std():.3f}")
    if normal:
        t, p = stats.ttest_ind(sz, hc, equal_var=False)
        d = pg.compute_effsize(sz, hc, paired=False, eftype='cohen')
        print(f"{device} {metric}: t-test p={p:.4f}, Cohen's d={d:.3f}")
    else:
        u, p = stats.mannwhitneyu(sz, hc, alternative='two-sided')
        d = pg.compute_effsize(sz, hc, paired=False, eftype='cles')
        print(f"{device} {metric}: Mann‑Whitney U p={p:.4f}, CLES={d:.3f}")


print('=' * 5, ' group diff ', '=' * 5)
for metric in ['dur', 'vel']:
    for device in ['eyelink', 'phone']:
        for para_id in [0, 1]:
            analyze_group_diff_by_para(metric, device, para_id)

print('=' * 5, ' para corr ', '=' * 5)
for metric in ['dur', 'vel']:
    for device in ['eyelink', 'phone']:
        for para_id in [0, 1]:
            analyze_para(metric, para_id)

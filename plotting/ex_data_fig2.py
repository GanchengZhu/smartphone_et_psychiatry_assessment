import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

from schizophrenia_detection import sz_dir
from schizophrenia_detection.variable_map import sz_feature_name_map


# ===== 设置Nature子刊风格 =====
def set_nature_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'font.size': 6,
        'axes.titlesize': 6,
        'axes.labelsize': 6,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'figure.figsize': (10, 8.5),
        'figure.dpi': 600,
        'axes.linewidth': 1,
        'grid.linewidth': 1,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'legend.frameon': False,
        'legend.title_fontsize': 6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,
        'axes.edgecolor': '.15',
        'grid.color': '.9',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })


set_nature_style()

# ===== 读取数据 =====
eyelink_path = f'{sz_dir}/features/data_eyelink'
phone_path = f'{sz_dir}/features/data_phone'


def load_and_merge_batches(base_path):
    df0 = pd.read_excel(f"{base_path}/batch_0.xlsx")
    df1 = pd.read_excel(f"{base_path}/batch_1.xlsx")
    return pd.concat([df0, df1], ignore_index=True)


df_eyelink = load_and_merge_batches(eyelink_path)
df_phone = load_and_merge_batches(phone_path)

# ===== 匹配subj_id并筛选公共特征 =====
common_ids = set(df_eyelink['subj_id']) & set(df_phone['subj_id'])
df_eyelink = df_eyelink[df_eyelink['subj_id'].isin(common_ids)].sort_values('subj_id').reset_index(drop=True)
df_phone = df_phone[df_phone['subj_id'].isin(common_ids)].sort_values('subj_id').reset_index(drop=True)

features_eyelink = df_eyelink.drop(columns=['subj_id'])
features_phone = df_phone.drop(columns=['subj_id'])

common_features = [f for f in (set(features_eyelink.columns) & set(features_phone.columns))]
# if 'q25' not in f and 'q75' not in f]

features_eyelink = features_eyelink[common_features]
features_phone = features_phone[common_features]

device_label_map = {
    'phone': 'iPhone',
    'eyelink': 'EyeLink'
}


# ===== 按任务类型排序特征 =====
def get_task_order(feature):
    """为特征分配排序优先级"""
    prefix = feature.split('_')[0]
    # 排序顺序: FS → FV → SP
    order_dict = {'fs': 0, 'fv': 1, 'sp': 2}
    return order_dict.get(prefix, 3), feature  # 返回元组用于排序


# 获取排序后的特征列表
sorted_features = sorted(common_features, key=get_task_order)

# ===== 计算相关系数矩阵 =====
corr_matrix = pd.DataFrame(
    index=sorted_features,
    columns=sorted_features,
    dtype=float
)


for i, eyelink_feat in enumerate(sorted_features):
    for j, phone_feat in enumerate(sorted_features):
        task_i = eyelink_feat.split('_')[0]
        task_j = phone_feat.split('_')[0]

        # 仅计算相同任务间的相关性
        if task_i == task_j:
            corr = features_eyelink[eyelink_feat].corr(features_phone[phone_feat])
            corr_matrix.iloc[i, j] = corr
        else:
            corr_matrix.iloc[i, j] = np.nan

# ===== 计算任务边界 =====
task_boundaries = {}
current_task = ""
start_idx = 0
feature_tasks = [f.split('_')[0] for f in sorted_features]

for i, task in enumerate(feature_tasks):
    if task != current_task:
        if current_task:  # 保存前一个任务的边界
            task_boundaries[current_task] = (start_idx, i)
        current_task = task
        start_idx = i
task_boundaries[current_task] = (start_idx, len(sorted_features))  # 最后一个任务

# ===== 对特征名进行映射替换 =====
corr_matrix.index = [sz_feature_name_map.get(f, f) for f in corr_matrix.index]
corr_matrix.columns = [sz_feature_name_map.get(f, f) for f in corr_matrix.columns]

# ===== 绘制热力图 =====
plt.figure()
ax = sns.heatmap(
    corr_matrix.astype(float),
    cmap="vlag",
    center=0,
    square=False,
    linewidths=0.7,
    cbar_kws={"label": "Pearson Correlation"},
    xticklabels=True,
    yticklabels=True,
    vmin=-0.6, vmax=0.6  # 确保颜色范围一致
)

# 添加任务分隔线
line_width = 0.5
for task, (start, end) in task_boundaries.items():
    if start > 0:  # 垂直分隔线
        ax.vlines(start, 0, len(corr_matrix), colors='gray', linewidths=line_width, linestyle='--', alpha=0.7)
    if end < len(corr_matrix):  # 水平分隔线
        ax.hlines(end, 0, len(corr_matrix), colors='gray', linewidths=line_width, linestyle='--', alpha=0.7)

# # 添加任务区块标签
# task_labels = {
#     'fs': 'Free Viewing (FS)',
#     'fv': 'Face Viewing (FV)',
#     'sp': 'Smooth Pursuit (SP)'
# }
# text_offset = 3
# for task, (start, end) in task_boundaries.items():
#     mid = (start + end) / 2
#     # Y轴标签
#     ax.text(-text_offset, mid, task_labels[task],
#             rotation=90, va='center', ha='center', fontsize=7)
#     # X轴标签
#     ax.text(mid, len(corr_matrix) + text_offset / 2, task_labels[task],
#             rotation=0, va='center', ha='center', fontsize=7)

# 添加任务区块边框
for task, (start, end) in task_boundaries.items():
    rect = Rectangle((start, start), end - start, end - start,
                     linewidth=1, edgecolor='dimgray', facecolor='none', zorder=10)
    ax.add_patch(rect)

# 画对角线：对应热力图的坐标轴范围，画斜线
plt.plot(
    [-0.5, len(corr_matrix.columns) - 0.5],  # x轴从-0.5到最大列索引+0.5，和热力图网格边界对齐
    [-0.5, len(corr_matrix.index) - 0.5],  # y轴同理
    color='gray',
    linewidth=0.7,
    linestyle='--',
    alpha=0.7
)

plt.xlabel("Smartphone Features")
plt.ylabel("Eyelink Features")
plt.title("Feature Correlation Heatmap: Eyelink vs. Smartphone", fontsize=7, pad=12)

save_dir = f'{os.path.dirname(__file__)}/ex_fig_2'
os.makedirs(save_dir, exist_ok=True)
plt.tight_layout()
plt.savefig(f"{save_dir}/correlation_heatmap.tiff", dpi=600, format='tiff')
# plt.show()

# print(f'All feature correlation: mean = {np.mean(corr_list):.2f}, std = {np.std(corr_list):.2f}')

import os.path

import numpy as np
import pandas as pd

from schizophrenia_detection import sz_dir
from schizophrenia_detection.variable_map import sz_feature_name_map
from utlis import get_stats_formatted

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

# ==== 加载精神分裂症数据 ====
sz_meta_file = f'{sz_dir}/meta_data/meta_data_release.xlsx'
sz_meta = pd.concat([pd.read_excel(sz_meta_file, sheet_name=f"batch_{i}") for i in [0, 1]])
sz_meta['id'] = sz_meta['id'].astype(str).str.lower()
sz_group_mapping = sz_meta.set_index('id')['sz'].to_dict()

df_eyelink['subj_id'] = df_eyelink['subj_id'].astype(str).str.lower()
df_phone['subj_id'] = df_phone['subj_id'].astype(str).str.lower()
df_eyelink.loc[:, 'label'] = df_eyelink['subj_id'].map(sz_group_mapping).astype(int)
df_phone.loc[:, 'label'] = df_phone['subj_id'].map(sz_group_mapping).astype(int)

# 创建合并的表格
combined_results = []

for feature in sorted_features:
    eyelink_row = get_stats_formatted(df_eyelink, feature)
    phone_row = get_stats_formatted(df_phone, feature)

    combined_row = [sz_feature_name_map.get(feature, feature)]
    combined_row.extend(eyelink_row)  # EyeLink results
    combined_row.extend(phone_row)  # Phone results

    combined_results.append(combined_row)

# 创建列名列表
columns = [
    'Feature',
    'EyeLink HC (Mean±95%CI)', 'EyeLinkSZ (Mean±95%CI)',
    'EyeLink t', 'EyeLink p', "EyeLink Cohen's d", "EyeLink t CI",
    'Phone HC (Mean±95%CI)', 'Phone SZ (Mean±95%CI)',
    'Phone t', 'Phone p', "Phone Cohen's d", "Phone Cohen's CI"
]

df_combined = pd.DataFrame(combined_results, columns=columns)

phone_t = pd.to_numeric(df_combined['Phone t'], errors='coerce')
eyelink_t = pd.to_numeric(df_combined['EyeLink t'], errors='coerce')

n_device_t_opposite = np.sum(
    phone_t * eyelink_t < 0
)
conflict_mask = (phone_t * eyelink_t < 0)
conflicting_features = df_combined[conflict_mask]
print(f"Number of features with opposite t-value directions across devices: {conflict_mask.sum()}")
print(conflicting_features[['Feature', 'Phone t', 'Phone p', 'EyeLink t', 'EyeLink p']])

# 写入 Excel
output_dir = f'{os.path.dirname(__file__)}/ex_table_2'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'eyelink_phone_combined_stats_ttest.xlsx')

with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    # 先写入数据
    df_combined.to_excel(writer, index=False, sheet_name='Combined Stats')

    # 获取 workbook 和 worksheet 对象
    workbook = writer.book
    worksheet = writer.sheets['Combined Stats']

    # 设置列宽
    worksheet.set_column(0, 0, 25)  # Feature列
    worksheet.set_column(1, 8, 15)  # EyeLink列
    worksheet.set_column(9, 16, 15)  # Phone列

    # 添加标题格式
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'align': 'center',
        'valign': 'vcenter',
        'border': 1
    })

    # 添加设备标题格式
    device_format = workbook.add_format({
        'bold': True,
        'align': 'center',
        'valign': 'vcenter',
        'border': 1,
        'bg_color': '#D3D3D3'  # 浅灰色背景
    })

    # 写入设备标题
    worksheet.write(0, 1, 'EyeLink', device_format)
    worksheet.merge_range(0, 1, 0, 8, 'EyeLink', device_format)

    worksheet.write(0, 9, 'Phone', device_format)
    worksheet.merge_range(0, 9, 0, 16, 'Phone', device_format)

    # 应用标题格式到列名
    for col_num, value in enumerate(df_combined.columns.values):
        worksheet.write(1, col_num, value, header_format)

print(f"Results saved to: {output_path}")

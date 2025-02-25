import pandas as pd

# 读取Excel数据集
data = pd.read_excel('data_1221.xlsx')

# 提取生源地区数据并去重
source_areas = data['生源地区'].unique()

# 创建生源地编号字典
source_area_dict = {area: i for i, area in enumerate(source_areas, 1)}

# 新增一列"生源地编号"，将生源地区映射为编号
data['生源地编号'] = data['生源地区'].map(source_area_dict)

# 保存带有生源地编号的数据为Excel文件
data.to_excel('data_108.xlsx', index=False)
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# 读取数据集
data = pd.read_excel('data_108.xlsx')

# 定义自变量和因变量
x = data[['性别', '民族', '政治面貌','生源地编号', '是否低保', '是否单亲家庭子女', '是否低收入家庭', '家庭人均年收入', '家庭人口数', '劳动力人口数', '困难等级']]
y = data['绩点成绩']

# 对分类变量进行独热编码
x = pd.get_dummies(x)

# 合并自变量和因变量
data = pd.concat([x, y], axis=1)

# 建立方差分析模型
model = ols('绩点成绩 ~ 性别 + 民族 + 政治面貌 + 生源地编号 + 是否低保 + 是否单亲家庭子女 + 是否低收入家庭 + 家庭人均年收入 + 家庭人口数 + 劳动力人口数 + 困难等级', data=data).fit()
anova_table = sm.stats.anova_lm(model)

# 打印方差分析结果
print(anova_table)
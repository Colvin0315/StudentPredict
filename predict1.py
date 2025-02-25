import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# 读取数据
data = pd.read_excel('data_1221_2.xlsx')

# 删除学号和绩点成绩列
data = data.drop(columns=['学号', '绩点成绩'])

# 提取特征和标签
X = data.drop(columns=['结果'])
y = data['结果']

# 对类别特征进行编码
label_encoder = LabelEncoder()
X['性别'] = label_encoder.fit_transform(X['性别'])
X['民族'] = label_encoder.fit_transform(X['民族'])
X['政治面貌'] = label_encoder.fit_transform(X['政治面貌'])
X['是否低保'] = label_encoder.fit_transform(X['是否低保'])
X['是否单亲家庭子女'] = label_encoder.fit_transform(X['是否单亲家庭子女'])
X['是否低收入家庭'] = label_encoder.fit_transform(X['是否低收入家庭'])
X['困难等级'] = label_encoder.fit_transform(X['困难等级'])

# 特征选择：根据特征重要性选择
# 假设你已经通过 RandomForest 得到特征重要性，然后筛选出最重要的特征
important_features = ['困难等级', '政治面貌', '民族', '是否低收入家庭', '性别', '是否单亲家庭子女', '是否低保']
X = X[important_features]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from imblearn.over_sampling import ADASYN

from imblearn.combine import SMOTEENN

# 使用 SMOTE + ENN（编辑最近邻）结合方法
smote_enn = SMOTEENN(sampling_strategy=1.0, random_state=42)
X_train_resampled, y_train_resampled = smote_enn.fit_resample(X_train, y_train)


# 随机森林模型训练
# 使用GridSearchCV进行超参数优化
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'random_state': [42]
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_resampled, y_train_resampled)

# 输出最佳参数
print("Best parameters found for Random Forest: ", grid_search.best_params_)

# 使用最佳参数的随机森林进行预测
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)

# 输出评估结果
print("Random Forest Accuracy: ", accuracy_score(y_test, y_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred))

# 预测某个特定样本
sample = np.array([1, 0, 0, 1, 1, 0, 0])  # 示例样本，填入你的特征值
sample = sample.reshape(1, -1)

# 使用最优随机森林进行预测
predicted = best_rf.predict(sample)
print("预测结果: ", predicted)


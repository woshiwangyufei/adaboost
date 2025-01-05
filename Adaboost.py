import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.tree import DecisionTreeRegressor
from scipy import stats
import seaborn as sns
import matplotlib
import joblib

# 读取数据
df = pd.read_csv('C:\\Users\\wjf18\\Desktop\\毕业论文\\工作一容量剩余容量\\剩余容量\\数据集\\9.21rl.csv')

# 分离非数值列
non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
numeric_df = df.drop(columns=non_numeric_columns)

# 移除全空的列
numeric_df = numeric_df.dropna(axis=1, how='all')

# 处理缺失值：使用K近邻填补
imputer = KNNImputer(n_neighbors=5)
numeric_df_imputed = imputer.fit_transform(numeric_df)
numeric_df = pd.DataFrame(numeric_df_imputed, columns=numeric_df.columns)

# 将非数值列添加回去
df = pd.concat([numeric_df, df[non_numeric_columns].reset_index(drop=True)], axis=1)

# 特征选择
features = ["AS",'IS',"DC","RP","AC","N",'IC']
X = df[features]
y = df['RC']

# 数据集拆分
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=10)

# 数据标准化
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# 超参数搜索
param_grid = {
    'n_estimators': [1200],
    'learning_rate': [0.1],
    'loss': ['linear', 'square'],
    'estimator__max_depth': [3],
    'estimator__min_samples_split': [2],
    'estimator__min_samples_leaf': [3],
}

n_splits = 10
seed = 5
k_fold = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
base_estimator = DecisionTreeRegressor()
Ada_regressor = AdaBoostRegressor(estimator=base_estimator)
model = GridSearchCV(estimator=Ada_regressor, scoring='r2', param_grid=param_grid, cv=k_fold, verbose=3, n_jobs=-1, return_train_score=True)
model.fit(X_train_std, y_train)
print(model.cv_results_)
df_result = pd.DataFrame(model.cv_results_)
df_result.to_csv('grid_search_results.csv', index=False)

# 获取最佳模型
best_model = model.best_estimator_
y_train_hat = best_model.predict(X_train_std)
y_test_hat = best_model.predict(X_test_std)

# 绘制预测结果图
fontsize = 12
plt.figure(figsize=(9, 8))
plt.style.use('default')
plt.rc('xtick', labelsize=fontsize)
plt.rc('ytick', labelsize=fontsize)
plt.rcParams['font.family'] = "Arial"
a = plt.scatter(y_train, y_train_hat, s=25, c='#b2df8a')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'k:', lw=1.5)
plt.xlabel('Observation', fontsize=fontsize)
plt.ylabel('Prediction', fontsize=fontsize)
plt.tick_params(direction='in')
plt.title('Train RMSE: {:.2e}, Test RMSE: {:.2e}'.format(np.sqrt(mean_squared_error(y_train, y_train_hat)), np.sqrt(mean_squared_error(y_test, y_test_hat))), fontsize=fontsize)
b = plt.scatter(y_test, y_test_hat, s=25, c='#1f78b4')
plt.legend((a, b), ('Train', 'Test'), fontsize=fontsize, handletextpad=0.1, borderpad=0.1)
plt.tight_layout()

plt.show()

# 保存预测结果
temp1 = np.c_[y_test, y_test_hat]
temp2 = np.c_[y_train, y_train_hat]
np.savetxt('C:\\Users\\wjf18\\Desktop\\毕业论文\\工作一容量剩余容量\\剩余容量\\结果\\adaboost.5.csv', temp1, delimiter=',')
np.savetxt('C:\\Users\\wjf18\\Desktop\\毕业论文\\工作一容量剩余容量\\剩余容量\\结果\\adaboost.6.csv', temp2, delimiter=',')
joblib.dump(model, 'AdaBoost_model.pkl')
# 打印评价指标
print('r (Train):', stats.pearsonr(y_train, y_train_hat))
print('r (Test):', stats.pearsonr(y_test, y_test_hat))
print('AdaBoost在训练集的决定系数R2: %.3f' % r2_score(y_train, y_train_hat))
print('AdaBoost在训练集的均方根误差RMSE: %.3f' % np.sqrt(mean_squared_error(y_train, y_train_hat)))
print('AdaBoost在训练集的平均绝对误差MAE: %.3f' % mean_absolute_error(y_train, y_train_hat))
print('AdaBoost在训练集的平均绝对百分误差MAPE: %.3f' % mean_absolute_percentage_error(y_train, y_train_hat))
print('AdaBoost在测试集的决定系数R2: %.3f' % r2_score(y_test, y_test_hat))
print('AdaBoost在测试集均方根误差RMSE: %.3f' % np.sqrt(mean_squared_error(y_test, y_test_hat)))
print('AdaBoost在测试集的平均绝对误差MAE: %.3f' % mean_absolute_error(y_test, y_test_hat))
print('AdaBoost在测试集的平均绝对百分误差MAPE: %.3f' % mean_absolute_percentage_error(y_test, y_test_hat))
print('参数的最佳取值：{0}'.format(model.best_params_))

# 加载模型
loaded_model = joblib.load('AdaBoost_model.pkl')

# 导入新数据
new_data = pd.read_csv('C:\\PythonProject\\RL\\combinations.csv')

# 特征处理
X_new = new_data[features]
X_new_std = scaler.transform(X_new)  # 使用训练数据的标准化参数进行标准化

# 预测
predictions = loaded_model.predict(X_new_std)

# 打印预测结果
result_df = pd.DataFrame({'Prediction': predictions})
result_df.to_csv('C:\\Users\\wjf18\\Desktop\\毕业论文\\容量剩余容量模型\\剩余容量\\验证数据\\结果\\predictions2.csv', index=False)

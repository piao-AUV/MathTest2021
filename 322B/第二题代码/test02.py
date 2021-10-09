#考虑到不少特征与房价并无强相关性，若带入所有数据可能会对模型训练引入噪音，
# 故而对所有模型，学生将对全数据集与经特征提取过的数据集分别套用相同模型，
# 用rse、rae、r2_score作为evaluation metrics（以r2_score为主）。

'''导入库'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,explained_variance_score,r2_score
from sklearn.svm import LinearSVR
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,ExtraTreesRegressor,RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV 
from statistics import mean, median
from sklearn import ensemble
from matplotlib.colors import ListedColormap
import logging
import logging.config
from sklearn.metrics import mean_squared_log_error
from sklearn.inspection import permutation_importance
from sklearn import datasets

'''加载模型以及提前设置的一些超参数'''
class EstimatorSelectionHelper:
    # 初始化, 加载模型以及提前设置的一些超参数
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.best = {}
    # 对每个模型的每组超参数都进行交叉验证
    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False):
        for key in self.keys:
            print("Running GridSearchCV for %s." % key)
            model = self.models[key]
            params = self.params[key]
            gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                              verbose=verbose, scoring=scoring, refit=refit,
                              return_train_score=True)
            gs.fit(X,y)
            self.best[key] = {'score':gs.best_score_,'params':gs.best_params_}
            self.grid_searches[key] = gs    
    # 对结果进行统计
    def score_summary(self, sort_by='mean_score'):
        def row(key, scores, params):
            d = {
                 'estimator': key,
                 'min_score': min(scores),
                 'max_score': max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores),
            }
            return pd.Series({**params,**d})

        rows = []
        for k in self.grid_searches:
            print(k)
            params = self.grid_searches[k].cv_results_['params']
            scores = []
            for i in range(self.grid_searches[k].cv):
                key = "split{}_test_score".format(i)
                r = self.grid_searches[k].cv_results_[key]        
                scores.append(r.reshape(len(params),1))

            all_scores = np.hstack(scores)
            for p, s in zip(params,all_scores):
                rows.append((row(k, s, p)))

        df = pd.concat(rows, axis=1).T.sort_values([sort_by], ascending=False)

        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]

        return df[columns]
    # 最优超参数.
    def best_params(self):
        return self.best


'''读取.CSV文件及处理'''
# boston = datasets.load_boston()
# diabetes = datasets.load_diabetes()
 
my_matrix = pd.read_csv("./bostonh.csv")
#对于矩阵而言，将矩阵倒数第一列之前的数值给了XPCA_（输入数据），将矩阵大最后一列的数值给了y（标签）
X, y = my_matrix.iloc[:,:-1].values,my_matrix.iloc[:,-1].values
#利用train_test_split方法，将X,y随机划分问，训练集（X_train），训练集标签（X_test），测试卷（y_train），
#测试集标签（y_test），安训练集：测试集=7:3的
#概率划分，到此步骤，可以直接对数据进行处理
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

#np.column_stack将两个矩阵进行组合连接
train= np.column_stack((X_train,y_train))
test = np.column_stack((X_test, y_test))
# print(my_matrix[1])
#数据预处理
Stand_X = StandardScaler()  # 特征进行标准化
Stand_Y = StandardScaler()  # 标签也是数值，也需要进行标准化
X_train = Stand_X.fit_transform(X_train)
X_test = Stand_X.transform(X_test)
y_train = Stand_Y.fit_transform(y_train.reshape(-1,1)) # reshape(-1,1)指将它转化为1列，行自动确定
y_test = Stand_Y.transform(y_test.reshape(-1,1)) 

'''成分方差，看降维保留了多少差异性'''
X_std=StandardScaler().fit_transform(X_train)
pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,7,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')#主成分的方差,用于量化了这些方向的重要程度
plt.savefig('Components variance')

#系数选择5，核密度估计(kernel density estimation)是在概率论中用来估计未知的密度函数，
# 属于非参数检验方法之一。通过核密度估计图可以比较直观的看出数据样本本身的分布特征。

sklearn_pca=PCA(n_components=5)
X_Train=sklearn_pca.fit_transform(X_std)

# '''生成表格对比数据'''
# models2 = {}
# models2["Linear"]        = linear_model.LinearRegression()
# models2["Ridge"]        = linear_model.Ridge(alpha=0.01)
# models2["Lasso"]         = linear_model.Lasso()
# models2["ElasticNet"]    = ElasticNet()
# models2["KNN"]           = KNeighborsRegressor() #K-近邻算法
# models2["DecisionTree"]  = DecisionTreeRegressor() #决策树
# models2["SVR"]           = SVR()
# models2["AdaBoost"]      = AdaBoostRegressor()
# models2["GradientBoost"] = GradientBoostingRegressor()
# models2["RandomForest"]  = RandomForestRegressor()
# models2["ExtraTrees"]    = ExtraTreesRegressor()

# models1 = {
#     'SVR': SVR(),
#     'KNeighborsRegressor': KNeighborsRegressor(),
#     'DecisionTreeRegressor': DecisionTreeRegressor(),
#     'RandomForestRegressor': RandomForestRegressor(),
#     "GradientBoostingRegressor":GradientBoostingRegressor(),
#     "AdaBoostRegressor":AdaBoostRegressor(),
#     "ElasticNet":ElasticNet(),
#     "ExtraTreesRegressor":ExtraTreesRegressor()
# }

# params1 = {
#     'SVR': {'kernel':('linear', 'rbf'), 'C':[1, 2, 4], 'gamma':[0.125, 0.25, 0.5 ,1, 2, 4]},
#     'KNeighborsRegressor': {'weights': ['uniform', 'distance'],
#                              'n_neighbors': range(2,100)
#                              },
#     'DecisionTreeRegressor':  {'max_features': ['sqrt', 'log2', None],
#                              'max_depth': range(2,1000),
#                              },
#     'RandomForestRegressor': {
#         'min_samples_split': list((3,6,9)),'n_estimators':list((10,50,100))},
#     "GradientBoostingRegressor":{'n_estimators':[100], 'learning_rate': [0.1], 'max_depth':[6],'min_samples_leaf':[3], 'max_features':[3]},

#     "AdaBoostRegressor":{'n_estimators': [50, 100], 'learning_rate' : [0.01,0.05,0.1,0.3,1],'loss' : ['linear', 'square', 'exponential']},
#     "ElasticNet": {"max_iter": [1, 5, 10],
#                       "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
#                       "l1_ratio": np.arange(0.0, 1.0, 0.1)},
#     "ExtraTreesRegressor":{
#         'max_features': range(1,4,1),
#     }
    
# }
# helper1 = EstimatorSelectionHelper(models1, params1)
# helper1.fit(X_train, y_train, scoring='r2', n_jobs=2)
# helper1.score_summary(sort_by='max_score')
# df_gridsearchcv_summary = helper1.score_summary()
# #print(type(df_gridsearchcv_summary.iloc[:,0:]))
# print(df_gridsearchcv_summary.iloc[:,0:])
# df_gridsearchcv_summary.iloc[:,0:].to_csv("Comparison.csv")

#综上所述，使用GradientBoostingRegressor: learning_rate=0.1, max_depth=6,
#max_features=13, min_samples_leaf=3, n_estimators=100并结合所有特征进行预测效果较佳。

'''选定最终模型与参数，进行模型评估'''
params = {'n_estimators': 100, 'max_depth': 6, 'max_features':3, 'min_samples_split': 2,
          'learning_rate': 0.1, 'min_samples_leaf': 3,'loss': 'ls'}
gbr = ensemble.GradientBoostingRegressor(**params)
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
# 估计器拟合训练数据
gbr.fit(X_train, y_train)
 
# 训练完的估计器对测试数据进行预测
y_pred = gbr.predict(X_test)

'''参数输出'''
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("平均绝对误差MAE：{}".format(mae))
print("均方误差MSE：{}".format(mse))
print("解释方差分EVS：{}".format(evs))
print("R2得分：{}".format(r2))

'''训练集与测试集偏差曲线'''
fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1,gbr.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')
fig.tight_layout()
#plt.show()
#plt.savefig('Pred_Deviance.png')

'''模型的训练测试收敛情况和各特征的重要性图示'''
feature_importance = gbr.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
fig = plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, np.array(my_matrix.columns)[sorted_idx])
plt.title('Feature Importance (MDI)')

result = permutation_importance (gbr, X_test, y_test, n_repeats=10,
                                random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(result.importances[sorted_idx].T,
            vert=False, labels=np.array(my_matrix.columns)[sorted_idx],whis=4)
plt.title("Permutation Importance (test set)")
fig.tight_layout()
#plt.show()
#plt.savefig("Importance.png")

'''模型评价，作图'''
t = np.arange(len(X_test))
plt.plot(t, y_test, color='red', linewidth=1.0, linestyle='-', label='y_test')
plt.plot(t, y_pred, color='green', linewidth=1.0, linestyle='-', label='y_pred')
plt.legend()
plt.grid(True)
#plt.show()
#plt.savefig('Test&Pred.png')
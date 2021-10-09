'''包的引入'''
import xlrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

'''数据读入与描述'''
names=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PRTATTO','B','LSTAT','MEDV']
data = pd.read_excel('./bostonh.xlsx',names=names)
# print(data)
# print(data.describe()) 

'''所有关系图的输出'''
sns.pairplot(data,kind="reg",diag_kind="kde", markers="+",palette="Accent_r")
sns.color_palette("hls", 8)
# plt.savefig('Sum Housing Price Relation')
plt.show()
# sns.pairplot(data,kind="reg",diag_kind="kde", markers="+",palette="husl",vars=["CRIM","MEDV"])
# plt.savefig('Housing Price & Crime.png')
# sns.pairplot(data,kind="reg",diag_kind="kde", markers="+",palette="husl",vars=["ZN","MEDV"])
# plt.savefig('Housing Price & Zone.png')
# sns.pairplot(data,kind="reg",diag_kind="kde", markers="+",palette="husl",vars=["INDUS","MEDV"])
# plt.savefig('Housing Price & Retail.png')
# sns.pairplot(data,kind="reg",diag_kind="kde", markers="+",palette="husl",vars=["CHAS","MEDV"])
# plt.savefig('Housing Price & ChasRiver.png')
# sns.pairplot(data,kind="reg",diag_kind="kde", markers="+",palette="husl",vars=["NOX","MEDV"])
# plt.savefig('Housing Price & NOX.png')
# sns.pairplot(data,kind="reg",diag_kind="kde", markers="+",palette="husl",vars=["RM","MEDV"])
# plt.savefig('Housing Price & Room.png')
# sns.pairplot(data,kind="reg",diag_kind="kde", markers="+",palette="husl",vars=["AGE","MEDV"])
# plt.savefig('Housing Price & Individual.png')
# sns.pairplot(data,kind="reg",diag_kind="kde", markers="+",palette="husl",vars=["DIS","MEDV"])
# plt.savefig('Housing Price & Length.png')
# sns.pairplot(data,kind="reg",diag_kind="kde", markers="+",palette="husl",vars=["RAD","MEDV"])
# plt.savefig('Housing Price & Access.png')
# sns.pairplot(data,kind="reg",diag_kind="kde", markers="+",palette="husl",vars=["TAX","MEDV"])
# plt.savefig('Housing Price & Tax.png')
# sns.pairplot(data,kind="reg",diag_kind="kde", markers="+",palette="husl",vars=["PRTATTO","MEDV"])
# plt.savefig('Housing Price & Teacher.png')
# sns.pairplot(data,kind="reg",diag_kind="kde", markers="+",palette="husl",vars=["B","MEDV"])
# plt.savefig('Housing Price & Black.png')
# sns.pairplot(data,kind="reg",diag_kind="kde", markers="+",palette="husl",vars=["LSTAT","MEDV"])
# plt.savefig('Housing Price & Layer.png')

# '''热力图分布与输出'''
# corrmat = data.corr()
# fig, ax = plt.subplots(figsize = (18, 10))
# sns.heatmap(corrmat, annot = True, annot_kws={'size': 12},cmap="Wistia")
# #plt.show()
# plt.savefig('Housing Price Relation Heat')

# 其中，LSTAT与price的相关程度最高(r=-0.74)，其次是RM,PTRAIO(|r|>=0.5), TAX,NOX(|r|>=0.4)。
# 因此，房价可能与LSTAT(底层人口的百分比),RM(每个住宅的平均房间数),PTRATIO(各镇的师生比率),
# TAX(每$ 10,000的全值财产税率),NOX(一氧化氮的浓度)有一定但不算太强的相关性。
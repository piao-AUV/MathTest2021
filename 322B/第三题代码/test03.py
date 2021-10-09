import pandas as pd
from math import *
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.optimize import  fsolve
import math

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

'''规定了加权方式,生成判断矩阵的简单方法'''
def get_judgement_matrix(scores):
    '''
    get judgement matrix  according to personal score.
    :param scores: a list, the item is the score range 1 to 10 means the importance of each sub-indicator.
    :return: judgement matrix, item range 1 to 9.

    - more: in judgement matrix:
    1 means two sub-indicators are the same important.

    3 means the first sub-indicator is a little important than another one.

    5 means the first sub-indicator is apparently important than another one.

    7 means the first sub-indicator is strongly significant than another one.

    9 means the first sub-indicator is extremely significant than another one.

    and 2, 4, 6, 8 are in the middle degree.
    '''

    # 评分1——150
    length = len(scores)

    array = np.zeros((length, length))
    for i in range(0, length):
        for j in range(0, length):
            point1 = scores[i]
            point2 = scores[j]
            deta = point1 - point2
            if deta < 0:
                continue
            elif deta == 0 or deta == 1:
                array[i][j] = 1
                array[j][i] = 1
            else:
                array[i][j] = deta
                array[j][i] = 1 / deta

    return array

'''获得判断矩阵的最大特征值和对应的特征向量'''
def get_tezheng(array):
    '''
    get the max eigenvalue and eigenvector
    :param array: judgement matrix
    :return: max eigenvalue and the corresponding eigenvector
    '''
    # 获取最大特征值和对应的特征向量
    te_val, te_vector = np.linalg.eig(array)
    list1 = list(te_val)

    max_val = np.max(list1)
    index = list1.index(max_val)
    max_vector = te_vector[:, index]

    return max_val, max_vector

'''获取RI值,随机一致性指标之一,另一个是CI'''
def RImatrix(n):
    '''
    get RI value according the the order
    :param n: matrix order
    :return: Random consistency index RI of a n order matrix
    '''
    n1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    n2 = [0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59, 1.60]
    d = dict(zip(n1, n2))
    return d[n]

'''计算一致性,进行一致性检验'''
def consitstence(max_val, RI, n):
    '''
    use the CR indicator to test the consistency of a matrix.
    :param max_val: eigenvalue
    :param RI: Random consistency index
    :param n: matrix order
    :return: true or false, denotes whether it meat the validation of consistency
    '''
    CI = (max_val - n) / (n - 1)
    if RI == 0:

        return True
    else:
        CR = CI / RI
        if CR < 0.10:
            #print("hhhhhhhh")
            return True
        else:
            #print("ZZZZZZZZZZZ")
            return False


def minMax(array):
    result = []
    for x in array:
        x = float(x - np.min(array)) / (np.max(array) - np.min(array))
        result.append(x)
    return np.array(result)

'''最大特征值对应的特征向量的归一化'''
def normalize_vector(max_vector):
    '''
    normalize the vector, the sum of elements is 1.0
    :param max_vector: a eigenvector
    :return: normalized eigenvector
    '''
    vector = []
    for i in max_vector:
        vector.append(i.real)
    vector_after_normalization = []
    sum0 = np.sum(vector)
    for i in range(len(vector)):
        vector_after_normalization.append(vector[i] / sum0)
    vector_after_normalization = np.array(vector_after_normalization)

    return vector_after_normalization

'''综合以上，输入子指标的打分向量，得到重要性权重向量。'''
def get_weight(score):
    '''
    get weight vector according to personal score.
    :param score: a list, the item is the score range 1 to 10 means the importance of each sub-indicator.
    :return: a list, the item is the weight range 0.0 to 1.0.
    '''
    n = len(score)
    array = get_judgement_matrix(score)
    max_val, max_vector = get_tezheng(array)
    RI = RImatrix(n)
    if consitstence(max_val, RI, n) == True:
        feature_weight = normalize_vector(max_vector)
        return feature_weight
    else:
        return [1 / n] * n

def getScore(array, point1, point2):
    '''
    a normalization function based on Human psychological satisfaction
    :param array: list, element is indicator's original value
    :param point1: the left expectation point, a list, [x1,y1]
    :param point2: the right expectation point, a list, [x2,y2]
    :return: normalized array
    '''
    x1 = point1[0]
    x2 = point2[0]
    y1 = point1[1]
    y2 = point2[1]

    def f1(a):
        equation1 = 1 / (1 + math.exp(-a * x1)) - y1
        return equation1

    def f2(a):
        equation1 = 1 / (1 + math.exp(-a * x2)) - y2
        return equation1

    # 存储归一化后的值
    values = []

    for i in array:
        try:
            i=i[0]
        except:
            pass
        if i < x1:
            sol3_fsolve = fsolve(f1, [0])
            a = sol3_fsolve[0]
            value = 1 / (1 + math.exp(a * (i - 2 * x1)))
        elif x1 <= i and i <= x2:
            value = (i - x1) * (y2 - y1) / (x2 - x1) + y1
        else:
            sol3_fsolve = fsolve(f2, [0])
            a = sol3_fsolve[0]
            value = 1 / (1 + math.exp(-a * i))
        values.append(value)

    # plt.scatter(array, values)
    # plt.show()
    return values

def show_score(value, title=''):
    x = np.linspace(1, len(value) + 1, len(value))
    plt.scatter(x, value)
    plt.title(title)
    #plt.show()
    plt.savefig("score.png")


def result(dataDict):
    '''人群打分'''
    def Layer_score():
        # 底层人口的百分比
        #data = pd.read_excel('./bostonh.xlsx',names=names)
        df = pd.read_excel('bostonh.xlsx')
        LSTAT_score_array = df.iloc[:, 12].values
        # print('LSTAT_score_array')
        # print(LSTAT_score_array)
        # 底层人口的百分比分数
        #print(type(LSTAT_score_array))
        LSTAT_score_array.resize(506,1)
        #print(LSTAT_score_array)
        LSTAT_score = getScore(LSTAT_score_array[:, 0], [dataDict['layer[LSTAT_score_min]'], 0.7],
                                   [dataDict['layer[LSTAT_score_max]'], 0.3])
        #print(LSTAT_score_array)
        # 各镇的师生比率
        PTRATTO_score_array = df.iloc[:, 10].values
        PTRATTO_score_array.resize(506,1)
        #print(PTRATTO_score_array)
        # 各镇的师生比率分数
        PTRATTO_score = getScore(PTRATTO_score_array[:, 0], [dataDict['layer[PTRATTO_score_min]'], 0.8],
                                     [dataDict['layer[PTRATTO_score_max]'], 0.2])

        # 城镇的人均犯罪率
        CRIM_score_array = df.iloc[:, 0].values
        CRIM_score_array.resize(506,1)
        #print(CRIM_score_array)
        # 城镇的人均犯罪率分数
        CRIM_score  = getScore(CRIM_score_array[:, 0], [dataDict['layer[CRIM_score_min]'],0.95],
                         [dataDict['layer[CRIM_score_max]'],0.05])       
        
        # 综合考虑
        Layer_score = [dataDict['layer[LSTAT_score]'], dataDict['layer[PRTATTO_score]'],dataDict['layer[CRIM_score]']]
        layer_weight = get_weight(Layer_score)
        #print(type(layer_weight))
        score_values = []
        # layer_weight_list = []
        # for i in layer_weight:
        #     layer_weight_list.append(round(i))
        for i in range(0, len(PTRATTO_score_array)):
            a1 = LSTAT_score[i]
            a2 = PTRATTO_score[i]
            a3 = CRIM_score[i]
            # print(type([a1,a2,a3]))
            # print(type(layer_weight))
            score_value = np.convolve(layer_weight,[a1, a2, a3])
            # score_value = layer_weight*[a1, a2, a3]
            score_values.append(sum(score_value))

        show_score(score_values,'人群指数')
        return score_values
    
    '''环境打分'''
    def environmental_score():
        df = pd.read_csv('bostonh.csv', index_col=0)
        array = df.loc[:, ['INDUS','CHAS', 'NOX', 'DIS', 'RAD']].values

        # 每个镇的非零售业务英亩的比例打分
        INDUS_score = getScore(array[:, 0], [dataDict['environment[INDUS_score_min]'], 0.5],
                                       [dataDict['environment[INDUS_score_max]'], 0.5])

        # 查尔斯河虚拟变量打分
        CHAS_score = getScore(array[:, 1], [dataDict['environment[CHAS_score_min]'], 0.9],
                                    [dataDict['environment[CHAS_score_max]'], 0.1])

        # 一氧化氮的浓度
        NOX_score = getScore(array[:, 2], [dataDict['environment[NOX_score_min]'], 0.5],
                                       [dataDict['environment[NOX_score_max]'], 0.5])

        # 到五个波士顿就业中心的加权距离打分
        DIS_score = getScore(array[:, 3], [dataDict['environment[DIS_score_min]'], 0.8],
                                      [dataDict['environment[DIS_score_max]'], 0.2])

        # 径向公路通达性的指标打分
        RAD_score = getScore(array[:, 4], [dataDict['environment[RAD_score_min]'], 0.8],
                                       [dataDict['environment[RAD_score_max]'], 0.2])

        #综合五个因素
        environmental_score = [dataDict['environment[score_INDUS]'], 
                        dataDict['environment[score_CHAS]'],
                        dataDict['environment[score_NOX]'],
                        dataDict['environment[score_DIS]'],
                        dataDict['environment[score_RAD]']]
        # 交通各个因素的权重为：
        environmental_weight = get_weight(environmental_score)

        environmental_scores = []
        for i in range(0, len(array)):
            a1 = INDUS_score[i]
            a2 = CHAS_score[i]
            a3 = NOX_score[i]
            a4 = DIS_score[i]
            a5 = RAD_score[i]
            environment_value = np.convolve(environmental_weight ,[a1, a2, a3, a4, a5])
            environmental_scores.append(sum(environment_value))
        show_score(environmental_scores,'环境指数')
        return environmental_scores

    '''经济打分'''
    def price_score():
        
        df = pd.read_csv('bostonh.csv')

        # 每$10,000的全值财产税率
        TAX_score_array = df.iloc[:, 9].values
        #print(TAX_score_array)
        TAX_score_array.resize(506,1)
        TAX_score = getScore(TAX_score_array[:, 0], [dataDict['price[TAX_score_min]'], 0.2],
                                  [dataDict['price[TAX_score_max]'], 0.8])

        #拥有住房价值的中位数
        MEDA_score_array = df.iloc[:, 13].values
        MEDA_score_array.resize(506,1)
        MEDA_score = getScore( MEDA_score_array[:, 0], [dataDict['price[MEDA_score_min]'], 0.2],
                                   [dataDict['price[MEDA_score_max]'], 0.8])

        # 综合考虑
        price_score = [dataDict['price[TAX_score]'], dataDict['price[MEDA_score]']]
        price_weight = get_weight(price_score)

        price_values = []
        for i in range(0, len(TAX_score_array)):
            a1 = TAX_score[i]
            a2 = MEDA_score[i]
            price_value = np.convolve(price_weight,[a1, a2])
            price_values.append(sum(price_value))
        show_score(price_values,'经济支出')
        return price_values

    '''其余因素打分'''
    def muti_score():
        df = pd.read_csv('bostonh.csv', index_col=0)

        # 非裔美国人的比例
        B_score_array = df.iloc[:, 11].values
        B_score_array.resize(506,1)
        B_score = getScore(B_score_array[:, 0], [dataDict['muti[B_score_min]'], 0.5],
                                  [dataDict['muti[B_score_max]'], 0.5])

        # 1940年之前建造的自有住房的比例
        AGE_score_array = df.iloc[:, 6].values
        AGE_score_array.resize(506,1)
        AGE_score = getScore( AGE_score_array[:, 0], [dataDict['muti[AGE_score_min]'], 0.4],
                                   [dataDict['muti[AGE_score_max]'], 0.6])

        # 每个住宅的平均房间数
        RM_score_array = df.iloc[:, 5].values
        RM_score_array.resize(506,1)
        RM_score = getScore( AGE_score_array[:, 0], [dataDict['muti[RM_score_min]'], 0.5],
                                   [dataDict['muti[RM_score_max]'], 0.5])
       
        #大于25,000平方英尺的地块的住宅用地比例
        ZN_score_array = df.iloc[:, 1].values
        ZN_score_array.resize(506,1)
        ZN_score = getScore( ZN_score_array[:, 0], [dataDict['muti[ZN_score_min]'], 0.9],
                                   [dataDict['muti[ZN_score_max]'], 0.1])

        # 综合考虑
        muti_score = [dataDict['muti[B_score]'], dataDict['muti[AGE_score]'],
                      dataDict['muti[RM_score]'],dataDict['muti[ZN_score]']]
        muti_weight = get_weight(muti_score)

        muti_values = []
        for i in range(0, len(B_score_array)):
            a1 = B_score[i]
            a2 = AGE_score[i]
            a3 = RM_score[i]
            a4 = ZN_score[i]
            muti_value = np.convolve(muti_weight,[a1, a2, a3,a4])
            muti_values.append(sum(muti_value))
        show_score(muti_values,'其余因素')
        return muti_values


    ps = Layer_score()
    ts = environmental_score()
    cs = price_score()
    ms = muti_score()
    ps=np.array(ps).reshape(506,)

    V = [dataDict['final[final_price]'],
         dataDict['final[final_traffic]'],
         dataDict['final[final_community]'],
         dataDict['final[final_muti]']]
    W = []
    W = get_weight(V)

    M = []
    data = []
    df = pd.read_csv('bostonh.csv', index_col=0)
    #names = df.loc[:, ['楼盘名称']].values[:, 0]
    prices = df.loc[:, ['MEDV']].values[:, 0]
    for i in range(0, len(ps)):
        a1 = ps[i]
        a2 = ts[i]
        a3 = cs[i]
        a4 = ms[i]
        q = W * [a1, a2, a3, a4]
        m = sum(q)
        M.append(m)
        #name = names[i]
        price = prices[i]
        data.append([ price, a1, a2, a3, a4, m])
    df = pd.DataFrame(data, columns=['房价中位数','人文指数', '自然环境', '消费水平','楼盘综合水平', '宜居性'])
    df.to_excel('comprehensive evaluation.xlsx')
    return data


if __name__ == '__main__':
    '''
    you can mark your personal score in daraDict
    '''
    dataDict = {'final[final_price]': 8,
                'final[final_traffic]': 6,
                'final[final_community]': 5,
                'final[final_muti]': 3,
                'final[final_location]': 4,
                'layer[LSTAT_score_min]': 2,
                'layer[LSTAT_score_max]': 37,
                'layer[PTRATTO_score_min]': 12.5,
                'layer[PTRATTO_score_max]': 22,
                'layer[CRIM_score_min]':0.006,
                'layer[CRIM_score_max]':89,
                'environment[INDUS_score_min]': 0.5,
                'environment[INDUS_score_max]': 27.5,
                'environment[CHAS_score_min]': 0,
                'environment[CHAS_score_max]': 1,
                'environment[NOX_score_min]': 0.4,
                'environment[NOX_score_max]': 0.9,
                'environment[DIS_score_min]': 1,
                'environment[DIS_score_max]': 12,
                'environment[RAD_score_min]': 1,
                'environment[RAD_score_max]': 24,
                'price[TAX_score_min]': 187,
                'price[TAX_score_max]': 711,
                'price[MEDA_score_min]': 5,
                'price[MEDA_score_max]': 50,
                'muti[B_score_min]': 0.3,
                'muti[B_score_max]':396,
                'muti[AGE_score_min]':3,
                'muti[AGE_score_max]':100,
                'muti[RM_score_min]':3.6,
                'muti[RM_score_max]':8.8,
                'muti[ZN_score_min]':0,
                'muti[ZN_score_max]':100,
                }
    
    dataDict['layer[LSTAT_score]']=150
    dataDict['layer[PRTATTO_score]']=15
    dataDict['layer[CRIM_score]']=45
    dataDict['environment[score_INDUS]']=35
    dataDict['environment[score_CHAS]'] = 2
    dataDict['environment[score_NOX]']=30
    dataDict['environment[score_DIS]']=22.5
    dataDict['environment[score_RAD]']= 1
    dataDict['price[TAX_score]'] = 25
    dataDict['price[MEDA_score]'] = 150
    dataDict['muti[B_score]'] = 12.5
    dataDict['muti[AGE_score]'] = 17.5
    dataDict['muti[RM_score]'] = 140
    dataDict['muti[ZN_score]'] = 7.5

    data = result(dataDict)
    #print(data)

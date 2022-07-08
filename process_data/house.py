import numpy
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

'''
数据构成
#   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             1460 non-null   int64  
 1   MSSubClass     1460 non-null   int64  
 2   MSZoning       1460 non-null   object 
 3   LotFrontage    1201 non-null   float64
 4   LotArea        1460 non-null   int64  
 5   Street         1460 non-null   object 
 6   Alley          91 non-null     object 
 7   LotShape       1460 non-null   object 
 8   LandContour    1460 non-null   object 
 9   Utilities      1460 non-null   object 
 10  LotConfig      1460 non-null   object 
 11  LandSlope      1460 non-null   object 
 12  Neighborhood   1460 non-null   object 
 13  Condition1     1460 non-null   object 
 14  Condition2     1460 non-null   object 
 15  BldgType       1460 non-null   object 
 16  HouseStyle     1460 non-null   object 
 17  OverallQual    1460 non-null   int64  
 18  OverallCond    1460 non-null   int64  
 19  YearBuilt      1460 non-null   int64  
 20  YearRemodAdd   1460 non-null   int64  
 21  RoofStyle      1460 non-null   object 
 22  RoofMatl       1460 non-null   object 
 23  Exterior1st    1460 non-null   object 
 24  Exterior2nd    1460 non-null   object 
 25  MasVnrType     1452 non-null   object 
 26  MasVnrArea     1452 non-null   float64
 27  ExterQual      1460 non-null   object 
 28  ExterCond      1460 non-null   object 
 29  Foundation     1460 non-null   object 
 30  BsmtQual       1423 non-null   object 
 31  BsmtCond       1423 non-null   object 
 32  BsmtExposure   1422 non-null   object 
 33  BsmtFinType1   1423 non-null   object 
 34  BsmtFinSF1     1460 non-null   int64  
 35  BsmtFinType2   1422 non-null   object 
 36  BsmtFinSF2     1460 non-null   int64  
 37  BsmtUnfSF      1460 non-null   int64  
 38  TotalBsmtSF    1460 non-null   int64  
 39  Heating        1460 non-null   object 
 40  HeatingQC      1460 non-null   object 
 41  CentralAir     1460 non-null   object 
 42  Electrical     1459 non-null   object 
 43  1stFlrSF       1460 non-null   int64  
 44  2ndFlrSF       1460 non-null   int64  
 45  LowQualFinSF   1460 non-null   int64  
 46  GrLivArea      1460 non-null   int64  
 47  BsmtFullBath   1460 non-null   int64  
 48  BsmtHalfBath   1460 non-null   int64  
 49  FullBath       1460 non-null   int64  
 50  HalfBath       1460 non-null   int64  
 51  BedroomAbvGr   1460 non-null   int64  
 52  KitchenAbvGr   1460 non-null   int64  
 53  KitchenQual    1460 non-null   object 
 54  TotRmsAbvGrd   1460 non-null   int64  
 55  Functional     1460 non-null   object 
 56  Fireplaces     1460 non-null   int64  
 57  FireplaceQu    770 non-null    object 
 58  GarageType     1379 non-null   object 
 59  GarageYrBlt    1379 non-null   float64
 60  GarageFinish   1379 non-null   object 
 61  GarageCars     1460 non-null   int64  
 62  GarageArea     1460 non-null   int64  
 63  GarageQual     1379 non-null   object 
 64  GarageCond     1379 non-null   object 
 65  PavedDrive     1460 non-null   object 
 66  WoodDeckSF     1460 non-null   int64  
 67  OpenPorchSF    1460 non-null   int64  
 68  EnclosedPorch  1460 non-null   int64  
 69  3SsnPorch      1460 non-null   int64  
 70  ScreenPorch    1460 non-null   int64  
 71  PoolArea       1460 non-null   int64  
 72  PoolQC         7 non-null      object 
 73  Fence          281 non-null    object 
 74  MiscFeature    54 non-null     object 
 75  MiscVal        1460 non-null   int64  
 76  MoSold         1460 non-null   int64  
 77  YrSold         1460 non-null   int64  
 78  SaleType       1460 non-null   object 
 79  SaleCondition  1460 non-null   object 
 80  SalePrice      1460 non-null   int64  
dtypes: float64(3), int64(35), object(43)
memory usage: 924.0+ KB
'''

'''
各数值型属性与'SalePrice'的相关系数
Id              -0.021917
MSSubClass      -0.084284
LotFrontage      0.351799
LotArea          0.263843
OverallQual      0.790982
OverallCond     -0.077856
YearBuilt        0.522897
YearRemodAdd     0.507101
MasVnrArea       0.477493
BsmtFinSF1       0.386420
BsmtFinSF2      -0.011378
BsmtUnfSF        0.214479
TotalBsmtSF      0.613581
1stFlrSF         0.605852
2ndFlrSF         0.319334
LowQualFinSF    -0.025606
GrLivArea        0.708624
BsmtFullBath     0.227122
BsmtHalfBath    -0.016844
FullBath         0.560664
HalfBath         0.284108
BedroomAbvGr     0.168213
KitchenAbvGr    -0.135907
TotRmsAbvGrd     0.533723
Fireplaces       0.466929
GarageYrBlt      0.486362
GarageCars       0.640409
GarageArea       0.623431
WoodDeckSF       0.324413
OpenPorchSF      0.315856
EnclosedPorch   -0.128578
3SsnPorch        0.044584
ScreenPorch      0.111447
PoolArea         0.092404
MiscVal         -0.021190
MoSold           0.046432
YrSold          -0.028923
SalePrice        1.000000
Name: SalePrice, dtype: float64
'''


def get(normalization=0, weight_flag=0):
    '''
    预处理house数据集
    主要在以下几个方面做处理:①读入数据,标识离散/连续型变量;②处理缺省值NAN;③数据标准化;④分配各特征的权重。

    :param normalization:为1表示需要标准化
    :param weight_flag:为1表示需要分配权重
    :return: train_X, train_y, test_X, data_train, data_test, labels, labels_form, weight
    '''
    # 读入训练数据和测试数据
    data_train = pd.read_csv('./dataset/house/train.csv')
    types = np.array(data_train.dtypes)[1:-1]  # 所有特征的数据类型,不包括'Id'和'SalePrice'
    cont_pos = np.where(types != numpy.object)[0]  # 获得所有连续属性即数值类特征的index
    data_test = pd.read_csv('./dataset/house/test.csv')

    # 转换数据为numpy数组,train_X去除‘ID’、'saleprice',train_y即'saleprice'
    train_X = np.array(data_train.iloc[:, :])[:, 1:-1]
    train_y = np.array(data_train.iloc[:, -1])
    test_X = np.array(data_test.iloc[:, :])[:, 1:]

    # 将训练数据和测试数据拼接起来方便后续标准化
    X = np.concatenate([train_X, test_X], axis=0)

    train_num = len(train_X)

    # 获取特征名
    labels = list(data_train.columns)[1:-1]

    # 特征类型:0/1 离散/连续
    labels_form = np.zeros(len(labels))
    labels_form[cont_pos] = 1

    weight = None
    # 根据相关系数绝对值大小计算权重
    if weight_flag:
        corrmat = data_train.corr('pearson')  # 计算相关系数矩阵
        corrs = np.array(corrmat['SalePrice'])[1:-1]  # 提取和'SalePrice'之间的相关系数
        corrs = np.abs(corrs)
        cont_weight_ratio = len(cont_pos) / len(labels)  # 连续型变量总占比
        ratio = corrs / np.sum(corrs)  # 各连续型变量应被分配的权重比例

        weight = np.ones(len(labels)) / len(labels)
        weight[cont_pos] = cont_weight_ratio * ratio  # 暂未考虑离散型

    # 数据标准化
    if normalization:
        transfer = StandardScaler()
        # 对连续值进行标准化,即x=(x-u)/σ
        X[:, cont_pos] = transfer.fit_transform(X[:, cont_pos].astype(float))

    # 连续型特征缺省值处理,直接令为0即可。
    continue_X = X[:, cont_pos]
    for i in range(len(continue_X)):
        for j in range(len(continue_X[0])):
            if np.isnan(continue_X[i, j]):
                continue_X[i, j] = 0
    X[:, cont_pos] = continue_X

    # 离散型特征缺省值处理，将为nan的变为字符串'NAN',决策树在递归构建子树时可将其视为一个单独的子树。
    discret_pos = np.where(labels_form == 0)[0]
    discret_X = X[:, discret_pos]
    for i in range(len(discret_X)):
        for j in range(len(discret_X[0])):
            x = discret_X[i, j]
            if type(x) == float and np.isnan(x):
                discret_X[i, j] = 'NAN'
    X[:, discret_pos] = discret_X

    return np.concatenate([X[:train_num], np.expand_dims(train_y, axis=1)], axis=1), \
           train_y, X[train_num:], data_train, data_test, labels, labels_form.tolist(), None

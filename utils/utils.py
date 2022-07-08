import numpy as np


def split(x, j, value, type):
    '''
    将样本数据集x按第j个特征进行划分
    type为离散'discret':提取j列特征为value的样本,并去掉第j列特征。
    type为连续型:即'L','R';前者提取小于等于value的;后者则大于。不删除第j列。

    :param x: 实例样本
    :param j: 特征j
    :param value: 离散:value为划分依据;连续: type='L',split<='value';type='R'同理。
    :param type: 'discret','L','R';
    :return: 划分后的数据集x
    '''
    if type == 'discret':
        feature_j = x[:, j]
        index = np.where(feature_j == value)
        x = np.delete(x[index], j, axis=1)
        return x
    feature_j = x[:, j].astype(float)
    index = np.where(feature_j <= float(value)) if type == 'L' else np.where(feature_j > float(value))
    return x[index]


def cal_square_error(x, j, form, weight):
    '''
    计算按第j个特征划分的均方误差MSE

    :param x: 样本数据,x[:,-1]为y值
    :param j: 目标为第j个feature
    :param form: 0/1:离散/连续
    :param weight: j特征的权重
    :return:error,best_split_value;即平方误差和对应最优划分的value值。
    '''
    if form == 0:  # 离散
        unique_features = set(x[:, j])  # 去重离散特征值
        if len(unique_features) == 1:  # j离散特征只有一个值
            return 1e20, None
        total_error = 0
        for feature in unique_features:
            y = x[np.where(x[:, j] == feature), -1][0].astype(np.float)  # 找出特征值为feature的实例
            mean_y = np.sum(y) / len(y)  # 均值
            error = np.sum(y ** 2) - 2 * mean_y * np.sum(y) + len(y) * mean_y ** 2  # 即Σ(yi-c)^2
            total_error += error  # 累加

        total_error /= len(x)
        return total_error / weight, None
    # 连续
    best_split_value = None  # 最优的切分点
    list_x = x.tolist()
    list_x.sort(key=lambda example: float(example[j]))  # 按第j特征从小到大排序
    x = np.array(list_x)

    count = 0  # count记录不同的特征值的总数-1;当count为0时说明只有一个特征值。
    now_value = float(x[0, j])  # 当前切分点
    minn = 1e20  # 最小均方误差
    for i in range(1, len(x)):
        if now_value == float(x[i, j]):  # 去重
            continue
        count += 1
        total_error = 0
        # <=now_value的被划分为"L",即0-(i-1)
        y = x[:i, -1].astype(np.float)
        mean_y = np.sum(y) / len(y)
        error = np.sum(y ** 2) - 2 * mean_y * np.sum(y) + len(y) * mean_y ** 2
        total_error += error

        # 同理>now_value的被划分到"R",即i-(len(x)-1)
        y = x[i:, -1].astype(np.float)
        mean_y = np.sum(y) / len(y)
        error = np.sum(y ** 2) - 2 * mean_y * sum(y) + len(y) * mean_y ** 2
        total_error += error
        # 以now_value为切分点,计算均方误差
        total_error /= len(x)
        if total_error < minn:
            minn = total_error
            best_split_value = now_value
        # 下一个unique_value
        now_value = float(x[i, j])
    if count == 0:  # 该j连续特征只有一个值,不用划分,返回split_value为None。
        return 1e20, None
    return minn / weight, best_split_value


def get_best_feature(x, label_form, weight):
    '''
    对当前样本数据x获取最优的特征

    :param x: 样本数据,x[:,-1]为实例y值
    :param label_form: label_form[j]:0/1 , j特征 离散/连续
    :param weight:
    :return: j,split_value,min_error 最优划分特征,最优的切分点(若最优特征是离散型特征该值为None),最优特征对应的MSE
    '''
    min_error = 1e20  # 最小误差
    min_j = None  # 最小误差对应的特征j
    best_split_value = None  # 最优切分点
    for j in range(len(label_form)):
        error, split_value = cal_square_error(x, j, label_form[j], weight[j])
        if error < min_error:
            min_error, best_split_value, min_j = error, split_value, j
    return min_j, best_split_value, min_error

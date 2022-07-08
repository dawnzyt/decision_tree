import numpy as np

from reg_dec_tree import regression_decison_tree


# 随机森林的类
class random_forest():
    def __init__(self, X, labels, labels_form, weight=None, limit_depth=9, least_samples=8, flag=0, ratio=0.75, num=10):
        '''
        随机森林构造函数

        :param X:
        :param labels:
        :param labels_form:
        :param weight:
        :param limit_depth:
        :param least_samples:
        :param flag: 从X-flag均与决策回归树类的构造参数相同
        :param ratio: 该参数代表构造随机森林的一个决策回归树时,实例样本占总样本数量的比例。
        :param num: 随机森林中决策回归树的数目
        '''
        self.X = X
        self.labels = labels
        self.labels_form = labels_form
        self.limit_depth = limit_depth
        self.least_samples = min(least_samples, len(X))
        self.flag = flag
        self.weight = weight
        if weight == None:
            self.weight = np.ones(len(X[0]) - 1)
            self.weight = 1 / (len(X[0]) - 1) * self.weight
        self.weight = list(self.weight)
        self.construct(ratio, num)

    def construct(self, ratio=0.75, num=10):
        '''
        构造随机森林

        :param ratio: 样本占比
        :param num: 随机森林数目
        :return:
        '''
        self.Ts = []
        samples_num = int(ratio * len(self.X))
        for i in range(num):
            rdm = np.random.permutation(len(self.X))
            X = self.X[rdm[:samples_num]]
            T = regression_decison_tree(X, self.labels, self.labels_form, weight=self.weight,
                                        limit_depth=self.limit_depth, least_samples=self.least_samples, flag=self.flag)
            self.Ts.append(T)

    def query(self, x):
        '''
        随机森林预测,返回预测结果

        :param x: 输入特征向量,务必和实例样本的特征结构相同。
        :return:
        '''
        ans = 0
        for i in range(len(self.Ts)):
            T = self.Ts[i]
            ans += T.query(x) / len(self.Ts)
        return ans

# 回归决策树,支持离散属性、连续属性建树，缺省值处理，回归预测。
import numpy as np
from utils import utils


# 决策回归树
class regression_decison_tree():
    def __init__(self, X, labels, labels_form, weight=None, limit_depth=9, least_samples=8,flag=0):
        '''
        决策回归树类的构造函数

        :param X: 实例样本数据,(m,n)m个样本,0-(n-2)为特征数据;n-1为y值。
        :param labels: (n-1),特征名
        :param labels_form: (n-1),特征类型:0/1表示离散/连续
        :param weight:(n-1),各特征的权重
        :param limit_depth: 决策树限制深度
        :param least_samples: 决策树结点最少实例数
        :param flag: 为1表示进行预剪枝
        '''
        self.X = X
        self.labels = labels
        self.labels_form = labels_form
        self.limit_depth = limit_depth
        self.least_samples = min(least_samples, len(X))
        self.weight = weight
        if weight == None:# 默认均匀权重
            self.weight = np.ones(len(X[0]) - 1)
            self.weight = 1 / (len(X[0]) - 1) * self.weight
        self.weight = list(self.weight)
        self.T = self.build(X, labels, labels_form, self.weight, dep=0, flag=flag)

    def build(self, x, labels, label_form, weight, dep, flag=0):
        '''
        递归建立决策回归树

        :param x: 样本数据集(m,n);m个样本,特征n-1维,最后一维是样本的y值。
        :param labels:labels[j] 特征j的名字
        :param label_form: label_form[j]表示特征j的数据类型:0 离散; 1 连续。
        :param weight: weight[j]表示特征j的权重
        :param dep: 当前深度
        :param flag: flag=1表示进行预剪枝,当前特征是否划分,如果不划分均方误差更小就不划分。
        :return: 决策树T,叶子节点返回一个元组(预测值y,实例数目);非叶子节点返回的是一个嵌套字典。
        '''
        y = x[:, -1].tolist()
        # 只有一种值
        if len(y) == y.count(y[0]):
            return (y[0], len(x))
        # 根据深度剪枝
        if dep >= self.limit_depth:
            return (sum(y) / len(y), len(x))
        # 如果样本实例小于设定阈值,直接返回
        if len(y) < self.least_samples:
            return (sum(y) / len(y), len(x))
        # 选择划分该树的最好特征;若对应特征是连续值一并返回最优划分value(离散该变量为None)。
        best_j, best_spilt_value, error = utils.get_best_feature(x, label_form, weight)
        # 如果最优划分best_j=None,说明当前结点不能划分了,直接返回平均值(实际上是因为所有特征的值均只有一个,没有划分的必要了)。
        if best_j == None:
            return (sum(y) / len(y), len(x))

        # 预剪枝-若不划分评估结果更优则直接返回。
        if flag == 1:
            y = np.array(y).astype(np.float)
            mean_y = np.sum(y) / len(y)
            not_divide_error = np.sum(y ** 2) - 2 * mean_y * np.sum(y) + len(y) * mean_y ** 2
            # print(not_divide_error/len(y)/weight[best_j]-error)
            if not_divide_error / len(y) / weight[best_j] - 0.1 < error:
                return (mean_y, len(y))
        # 已选好最优特征,建立决策树
        feat_name = labels[best_j]

        if label_form[best_j] == 0:  # 离散情况
            # 嵌套字典作为决策树的数据结构;该节点的name为划分特征的label。
            T = {feat_name: {}, 'num': len(x)}
            unique_feat = set(x[:, best_j])

            for feat in unique_feat:
                sub_labels = labels[:]
                sub_label_form = label_form[:]
                sub_weigt = weight[:]
                del (sub_labels[best_j])
                del (sub_label_form[best_j])
                del (sub_weigt[best_j])
                T[feat_name][feat] = self.build(utils.split(x, best_j, feat, type='discret'), sub_labels,
                                                sub_label_form, sub_weigt, dep + 1, flag)
        else:  # 连续情况
            # 该节点的名称包含我们分割的特征label名称和该特征的最优分割value;在查询的时候split'<'即可使用。
            T_name = str(feat_name) + '<' + str(best_spilt_value)
            T = {T_name: {}, 'num': len(x)}
            T[T_name]['L'] = self.build(utils.split(x, best_j, best_spilt_value, type='L'),
                                        labels[:], label_form[:], weight, dep + 1, flag)
            T[T_name]['R'] = self.build(utils.split(x, best_j, best_spilt_value, type='R'),
                                        labels[:], label_form[:], weight, dep + 1, flag)
        return T

    def rt(self):
        '''
        返回决策树根节点self.T

        :return:
        '''
        return self.T

    def query(self, x):
        '''
        query函数,对外的接口，预测x的y值;
        务必使x的特征数据形式和self.X即实例训练样本的特征形式相同。

        :param x:
        :return:
        '''
        return self.Q(self.T, x, self.labels, self.labels_form)

    def Q(self, T, x, labels, label_form):
        '''
        Q:递归查询x的类别

        :param T: 决策树当前节点T
        :param x: 查询的特征向量
        :param labels: 特征name list
        :param label_form: list:0/1 表示特征 离散/连续
        :return: x的类别预测
        '''
        # 叶子结点
        if type(T).__name__ != 'dict':
            return T[0]
        key = list(T.keys())[0]  # list(T.keys())[1]即'num',T['num']记录了当前结点划分的实例数目
        son = T[key]  # son字典记录了key特征的各个特征值对应的子决策树。
        try:  # 连续特征
            feat_name, value = key.split(sep='<')  # 如果是离散特征,没有'<'报'ValueError'错
            j = labels.index(feat_name)
            which = 'L' if float(x[j]) <= float(value) else 'R'
            return self.Q(son[which], x, labels, label_form)
        except ValueError:  # 离散特征
            feat_name = key
            j = labels.index(feat_name)
            # 决策树没有某特征的一特征值的划分
            if son.get(x[j], None) == None:
                ans = 0
                choose_key = None
                maxx = 0
                # 从儿子中选择实例数目最多的返回
                for key in son.keys():
                    num = 0
                    if type(son[key]).__name__ == 'dict':  # 非叶子为嵌套字典
                        num = son[key]['num']
                    else:  # 叶子为元组(平均y,num)
                        num = son[key][1]
                    if num > maxx:
                        maxx = num
                        choose_key = key
                return self.Q(son[choose_key], x, labels, label_form)
                # # 所有划分特征值的平均数
                # for key in son.keys():
                #     ans += query(son[key], x, labels, label_form)
                # # 返回所有子树查询结果的平均值
                # return ans / len(son.keys())
            return self.Q(son[x[j]], x, labels, label_form)

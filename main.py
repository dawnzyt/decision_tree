import numpy as np
from random_forest import random_forest
import pandas as pd
from reg_dec_tree import regression_decison_tree
from process_data import house

if __name__ == '__main__':
    # 预处理数据
    train_X, train_y, test_X, data_train, data_test, labels, \
    labels_form, weight = house.get(normalization=1, weight_flag=1)

    # # 将房价取ln
    # train_X[:, -1] = np.log(train_X[:, -1].astype(np.float))

    # 构造决策回归树
    reg_T = regression_decison_tree(train_X, labels, labels_form, weight=weight, limit_depth=8, least_samples=8, flag=1)

    # # 构造随机森林
    # forest = random_forest(train_X, labels, labels_form, weight, limit_depth=5,
    #                        least_samples=5,flag=1, ratio=0.5, num=10)

    # 利用决策回归树预测测试数据
    pred_y = [reg_T.query(x) for x in test_X]

    # # 利用随机森林预测测试数据
    # pred_y = [forest.query(x) for x in test_X]

    # # 取指数
    # pred_y = np.exp(pred_y)

    # 写出预测测试数据的结果
    pred_sale = pd.DataFrame(pred_y, columns=['SalePrice'])
    submit = pd.concat([data_test['Id'], pred_sale], axis=1)
    submit.to_csv('regresion_tree_result.csv', index=False)

    # # 计算训练数据对数均方根误差。
    # RMSE = 0
    # for i in range(len(train_X)):
    #     logit = reg_T.query(train_X[i])
    #     RMSE += (log(logit, math.e) - log(train_y[i], math.e)) ** 2
    # RMSE = np.sqrt(RMSE / len(train_X))
    # print("训练用训练数据均方根误差为:", RMSE)

# 前言
本代码针对kaggle入门竞赛-**房价预测**给出了一种拓展的非二叉决策回归树的方法，支持离散数据和连续数据建树，并且引入了随机森林。本文档将介绍如何使用该代码进行相应的测试;同时在面对其它回归问题时如何使用本代码中定义的**决策回归树**和**随机森林**。

[比赛入口](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)

# 环境
 - python3.7
 - numpy

纯净的python环境即可

# 数据预处理

赛题数据已经在`dataset/house`目录下，`process_data`目录下的`house.py`的注释中包括了本赛题数据的组成和数据类型，同时该`house.py`中定义了对本赛题数据的预处理方法，包括数据标准化、权重分配。

# 决策回归树

决策回归树类的定义与赛题没有联系，其中`utils/utils.py`中定义了一些构造决策回归树类时使用的的一些方法和函数，`reg_dec_tree.py`中就定义了我们的**决策回归树类**。

- **构造一个决策回归树类：**
```buildoutcfg
reg_T = regression_decison_tree(train_X, labels, labels_form, weight=weight, limit_depth=9, least_samples=8,flag=1)
```

***注意：*** *其构造需要参数的要求、格式可以转到对应文件去查看，有详细的注释，这里不再一一说明。*

 - **预测一个输入的特征向量：**

```buildoutcfg
y=reg_T.query(x)
```
***注意：*** *务必保证x特征向量的格式和`train_X`样本的特征格式相同。*

 - **其它方法请自行查看**

# 随机森林

本项目中的随机森林是直接基于我编写的决策回归树来构造的。

 - **构建随机森林：**

```buildoutcfg
forest = random_forest(train_X, labels, labels_form, weight, limit_depth=5,
                           least_samples=5,flag=1, ratio=0.5, num=10)
```

***注意：*** *可以发现参数基本和`regression_decison_tree`类的构造一致，**ratio** 代表随机样本比例 **num** 代表随机森林的大小，即所包含决策回归树树的数目*

- **预测一个输入的特征向量：**

```buildoutcfg
y=forest.query(x)
```

# 房价预测实验

我们房价预测的实验在`main.py`中进行，实际上，只需要逐一调用我定义的这些类和方法便可完成本次实验，不过这里还是简单阐述一下具体步骤：

~~**1. 预处理数据**~~

```buildoutcfg
train_X, train_y, test_X, data_train, data_test, labels, labels_form, weight = house.get(normalization=1, weight_flag=1)
```
 - **返回参数解释**

`train_X` 我们训练的实例样本为(m,n)，其中第n列是目标y值，等于train_y，0-（n-2）列才是特征。

`labels` (n-1) n-1个特征的特征名

`labels_form` (n-1) n-1个特征的类别，`labels_form[j]`0/1表示特征j是离散/连续

`weight` n-1个特征的权重，如果想要均匀权重可以置为`None`

 - **调用参数**

`normalization` 1表示对训练样本的连续型数据进行标准化

`weight_flag` 1表示会根据各特征之间的关系对连续型特征合理分配权重，若为0,`weight`将为None

~~**2. 构造决策回归树或随机森林**~~

 - **构造决策回归树**

实际上，我们在构造决策回归树和随机森林时候所必须的参数就只有`train_X` 、`labels`和`labels_form`，其它都是拓展。
```buildoutcfg
reg_T = regression_decison_tree(train_X, labels, labels_form, weight=weight, limit_depth=9, least_samples=8,flag=1)
```
可改变的参数：

`limit_depth` 决策树限制的最深深度（根节点深度为0）

`least_samplse` 决策树一个中间结点最少包括的实例总数

`flag` flag=1表示进行预剪枝-验证是否需要构造子决策树

 - **构造随机森林**

```buildoutcfg
forest = random_forest(train_X, labels, labels_form, weight, limit_depth=5,
                           least_samples=5,flag=1, ratio=0.5, num=10)
```
可改变的参数：

`ratio` 代表构造随机森林中的决策回归树时的输入实例样本占总样本`train_X`的比例

`num` 代表随机森林的大小即决策回归树的数目

~~**3. 预测**~~

以决策回归树为例子，pred_y即预测的结果：
```buildoutcfg
pred_y = [forest.query(x) for x in test_X]
```
之后再将其保存为对应的`.csv`文件即可，如下：
```buildoutcfg
# 写出预测测试数据的结果
pred_sale = pd.DataFrame(pred_y, columns=['SalePrice'])
submit = pd.concat([data_test['Id'], pred_sale], axis=1)
submit.to_csv('regresion_tree_result.csv', index=False)
```

~~**4. 提交结果**~~

将对应的`regresion_tree_result.csv`提交到 [网址](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/submit) 中去即可。

# 其它实验

实际上要完成其它实验很简单，只需要完成相应数据的预处理，使其与我编写的决策回归树以及随机森林构造所需要的参数格式一致再调用相关接口即可，这里就不再过多赘述了。

# 补充

1. 在完成本次实验时，有一个拓展步骤是对训练样本的目标值取了对数，然后再以此建立决策树，最后预测的时候还需要取指数，这里我将其注释掉了，若需要可以自行修改。

2. 注意本决策回归树的构造是不支持样本中带有缺省值的，也就是说，必须提前在预处理部分将缺省值处理掉，本项目的`house.py`中是完成了缺省值的处理的，若需要将本项目应用到其它实验，请注意这一点。

3. `data_analysis.iypnb`是在kaggle中完成的数据分析的notebook，若需要查看具体内容可以到kaggle平台导入该notebook，同时注意要在kaggle导入本赛题的数据集。
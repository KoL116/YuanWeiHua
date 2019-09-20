# 决策树

  是一种非参数的有监督学习方法，它能够从一系列有特征和标签的数据中总结出决策规则，并用树状图的结构来呈现这些规则，以解决分类和回归问题。决策树算法容易理解，适用各种数据，在解决各种问题时都有良好表现，尤其是以树模型为核心的各种集成算法，在各个行业和领域都有广泛的应用，决策树算法的本质是一种图结构。



- 环境：anaconda、jupyter notebook、python2.7

- 项目库：

  ```
  from sklearn.datasets import load_iris
  ```

  ```
  sklearn.datasets模块主要提供了一些导入、在线下载及本地生成数据集的方法，可以通过dir或help命令查看，目前主要有三种形式：
      load_<dataset_name> 本地加载数据
      fetch_<dataset_name> 远程加载数据
      make_<dataset_name> 构造数据集
  Iris 鸢尾花数据集是一个经典数据集，在统计学习和机器学习领域都经常被用作示例。数据集内包含 3 类共 150 条记录，每类各 50 个数据，每条记录都有 4 项特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度，可以通过这4个特征预测鸢尾花卉属于（iris-setosa, iris-versicolour, iris-virginica）中的哪一品种。
  ```

	

	```
	from sklearn import tree
	```

	```
	”tree“模块包含五个类：
	tree.DecisionTreeClassifier 分类树、tree.DecisionTreeRegressor 回归树、tree.export_graphviz 将生成的决策树导出为DOT格式、tree.ExtraTreeClassifier 高随机版本分类树、tree.ExtraTreeRegressor 高随机版本回归树
	```
	
	```
	from sklearn.model_selection import train_test_split
	```
	
	```
	from sklearn.model_selection import cross_val_score
	```
	
	```
	from sklearn.model_selection import GridSearchCV
	```
	
	```
	model_selection包括交叉验证，微调模型的超参数以及学习曲线三部分。可用的API有拆分策略，参数优化的方法以及模型评估
	
	sklearn是机器学习中一个常用的python第三方模块，里面对一些常用的机器学习方法进行了封装，在进行机器学习任务时，并不需要每个人都实现所有的算法，只需要简单的调用sklearn里的模块就可以实现大多数机器学习任务。
	```
	
	```
	import graphviz
	```
	
	```
	GraphViz是一个开源的图像可视化的软件，是贝尔实验室开发的一个开源的工具包，它使用一个特定的DSL(领域特定语言): dot作为脚本语言，然后使用布局引擎来解析此脚本，并完成自动布局。graphviz提供丰富的导出格式，如常用的图片格式，SVG，PDF格式等。
	```
	
	```
	import matplotlib.pyplot as plt
	```
	
	```
	提供一个类似MATLAB的绘图框架。
	matplotlib.pyplot 是命令样式函数的集合，使matplotlib像MATLAB一样工作。 每个pyplot函数对图形进行一些更改：例如，创建图形，在图形中创建绘图区域，在绘图区域中绘制一些线条，用标签装饰图形等。
	在matplotlib.pyplot中，各种状态在函数调用中保留，以便跟踪当前图形和绘图区域等内容，并且绘图函数指向当前轴（请注意“轴”在此处以及在大多数位置 文档是指图形的轴部分，而不是多个轴的严格数学术语。
	```
	
	```
	import numpy as np
	```
	
	```
	NumPy是Python中科学计算的基础软件包。
	它是一个提供多了维数组对象，多种派生对象（如：掩码数组、矩阵）以及用于快速操作数组的函数及API，它包括数学、逻辑、数组形状变换、排序、选择、I/O 、离散傅立叶变换、基本线性代数、基本统计运算、随机模拟等等。NumPy包的核心是ndarray对象。它封装了python原生的同数据类型的n维数组，为了保证其性能优良，其中有许多操作都是代码在本地进行编译后执行的。
	```
	

	```
import pandas
Pandas是一个开源的，BSD许可的库，为Python编程语言提供高性能，易于使用的数据结构和数据分析工具。
	```

	```
	import mlxtend
Mlxtend是一个基于Python的开源项目，主要为日常处理数据科学相关的任务提供了一些工具和扩展。
	```



- 注：在anaconda prompt中执行pip install graphviz后，将graphviz文件夹路径添加至系统路径中



### 重要参数

- criterion：决策树找出最佳节点和最佳的分枝方法的指标叫做“不纯度”。通常来说，不纯度越低，决策树对训练集的拟合越好。Criterion参数决定不纯度的计算方法。
- random_state：用来设置分枝中的随机模式的参数，默认None，在高维度时随机性会表现更明显。
- splitter：用来控制决策树中的随机选项，有两种输入值，输入”best"，决策树在分枝时虽然随机，但是还是会优先选择更重要的特征进行分枝（重要性可以通过属性feature_importances_查看），输入“random"，决策树在分枝时会更加随机，树会因为含有更多的不必要信息而更深更大，并因这些不必要信息而降低对训练集的拟合。这也是防止过拟合的一种方
- max_depth：限制树的最大深度，超过设定深度的树枝全部剪掉。
- min_samples_leaf ：限定，一个节点在分枝后的每个子节点都必须包含至min_samples_leaf个训练样本，否则分枝就不会发生，或者，分枝会朝着满足每个子节点都包min_samples_leaf个样本的方向去发生。
- min_samples_split：限定，一个节点必须要包含至少min_samples_split个训练样本，这个节点才允许被分枝，否则分枝就不会发生。
- max_features：限制分枝时考虑的特征个数，超过限制个数的特征都会被舍弃
- min_impurity_decrease：限制信息增益的大小，信息增益小于设定数值的分枝不会发生。



#### LR回归

  LR回归是在线性回归模型的基础上，使用sigmoid函数，将线性模型w^t x的结果压缩到[0,1]之间，使其拥有概率意义。 其本质仍然是一个线性模型，实现相对简单。在广告计算和推荐系统中使用频率极高，是CTR预估模型的基本算法。同时，LR模型也是深度学习的基本组成单元。LR回归属于概率性判别式模型


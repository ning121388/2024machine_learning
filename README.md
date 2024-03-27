# machine learning

## 3.7

1. 机器学习是拟合数学模型，利用线性、非线性结合的方法去拟合。
2. 拟合的本质是复合函数

## 3.14

### 2.Linear Algebra

1. Ln正则化是将xi进行n次平方加和再开n次根号
2. 特征向量/特征值：是对矩阵和x乘积，***只有伸缩变换***，没有旋转变换
3. SVD：利用更小的矩阵（3个），降低矩阵的大小，进行压缩

### 3.Probability and statistics

1. 幸存者偏差：所见非真相
2. 各种分布的掌握，以高斯分布为基础
3. 贝叶斯理论及贝叶斯公式的应用（与条件概率类似）

## 3.21

### 4.Linear model

1. 欠拟合和过拟合：对于输入信息的 ***噪声*** 的拟合程度

2. Occam`s razor理论：采用先验知识对所获得的数据进行假设，采取最简单、最普遍的假设

3. 最小二乘法算最优解，对比向量解释，求e的向量积为0（垂直）最小![image-20240321170857150](C:\Users\13059\AppData\Roaming\Typora\typora-user-images\image-20240321170857150.png)

4. 梯度下降：按梯度大小进行更新，不会错过梯度为0的点
   $$
   J(\theta_0, \theta_1) = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{2N} \sum_{i=1}^{N} (f(x^{(i)})-y^{(i)})^2 =  \frac{1}{2N} \sum_{i=1}^{N} (\theta_0 + \theta_1 x^{(i)} - y^{(i)})^2
   $$
   对损失函数进行偏导，对参数进行偏导
   $$
   \frac{\partial J}{\partial \theta_0} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)})
   ;
         \frac{\partial J}{\partial \theta_1} = \frac{1}{N} \sum_{i=1}^{N} (\hat{y}^{(i)} - y^{(i)}) x_i
   $$
   得到最后的
   $$
   \theta_0 := \theta_0 - \alpha \frac{\partial J}{\partial \theta_0}; \theta_1 := \theta_1 - \alpha \frac{\partial J}{\partial \theta_1}
   $$

5. 罗杰斯特回归：采取激活函数对线性方程进行处理，调整参数仍然是线性部分
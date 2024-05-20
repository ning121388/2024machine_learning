import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def male_and_female(N_total=500):
    # 设置均值和标准差
    mean_male = 176
    mean_female = 164
    sigma_male = 8
    sigma_female = 6
    # 生成1000个随机数据点
    data_male = np.random.normal(mean_male, sigma_male, int(N_total*0.6))
    data_female = np.random.normal(mean_female, sigma_female, int(N_total*0.4))
    return data_male,data_female

# # 生成x轴的数据点
# x_f = np.linspace(130, 219, 200)
# x_m = np.linspace(130, 219, 200)
# # 计算对应的正态分布概率密度函数值
# y_m = norm.pdf(x_m, 176, 8)
# y_f = norm.pdf(x_f, 164, 6)
#
# # 绘制数据的直方图
# plt.plot(x_m, y_m, label=f'Normal Distribution ', color='blue')
# plt.plot(x_f, y_f, label=f'Normal Distribution ', color='red')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.show()

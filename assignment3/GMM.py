from height import male_and_female
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm
import statistics
def EM(data, max_epoch=1000):
    # 1 步
    n = len(data)
    # 计算中位数
    median_data = statistics.median(data)
    # 初始化参数
    mu1, mu2 = median_data+5, median_data-5  # 随机初始化均值
    sigma1_sq, sigma2_sq = np.random.rand(2) * 10 + 5  # 随机初始化方差
    pi1, pi2 = np.random.rand(2)
    pi1 /= (pi1 + pi2)  # 确保混合权重之和为1

    for _ in range(max_epoch):
        # 2 步
        gamma1 = pi1 * norm.pdf(data, mu1, np.sqrt(sigma1_sq))  #概率密度×混合权重
        gamma2 = pi2 * norm.pdf(data, mu2, np.sqrt(sigma2_sq))
        total_gamma = gamma1 + gamma2   #除以总值
        gamma1 /= total_gamma
        gamma2 /= total_gamma

        # 3 步
        mu1 = np.sum(gamma1 * data) / np.sum(gamma1)    #优化均值
        mu2 = np.sum(gamma2 * data) / np.sum(gamma2)
        sigma1_sq = np.sum(gamma1 * (data - mu1) ** 2) / np.sum(gamma1)     #优化方差
        sigma2_sq = np.sum(gamma2 * (data - mu2) ** 2) / np.sum(gamma2)
        pi1 = np.mean(gamma1)   #优化混合权重
        pi2 = np.mean(gamma2)

        # # 打印每次迭代的结果
        # print(
        #     f"Iteration {_ + 1}: mu1={mu1:.2f}, mu2={mu2:.2f}, sigma1^2={sigma1_sq:.2f}, sigma2^2={sigma2_sq:.2f}, pi1={pi1:.2f}")

    return np.array([mu1, mu2]), np.array([sigma1_sq, sigma2_sq]), np.array([pi1, pi2])

data_male,data_female = male_and_female(N_total= 1000)
data = np.concatenate([data_male, data_female])

# # 使用自己编写的EM函数
# means, sigmas, weights=EM(data)
# sigmas = np.sqrt(sigmas)
# print("GMM Means:", means)
# print("GMM Sigmas:", sigmas)
# print("GMM Weights:", weights)
# x = np.linspace(140, 210, 1000)
# y_m = norm.pdf(x, 176, 8)*0.6
# y_f = norm.pdf(x, 164, 6)*0.4
# y_m_p = norm.pdf(x, means[0], sigmas[0])*weights[0]
# y_f_p = norm.pdf(x, means[1], sigmas[1])*weights[1]
# total_pdf = norm.pdf(x, means[0], sigmas[0])*weights[0] + norm.pdf(x, means[1], sigmas[1])*weights[1]
# plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
# plt.plot(x, total_pdf, '-k', label='Personal PDF')
# plt.plot(x, y_m_p, 'b', label='pre male PDF')
# plt.plot(x, y_f_p, 'r', label='pre female PDF')
# plt.plot(x, y_m, '--b', label='real male PDF')
# plt.plot(x, y_f, '--r', label='real female PDF')
# plt.legend()
# plt.title("Personal EM Fit")
# plt.xlabel("Height (cm)")
# plt.ylabel("Density")
# plt.show()

# 使用包装好的GMM
gmm = GaussianMixture(n_components=2, max_iter=200, random_state=42)
gmm.fit(data.reshape(-1, 1))

# 获取GMM参数
means = gmm.means_.flatten()
sigmas = np.sqrt(gmm.covariances_.flatten())
weights = gmm.weights_.flatten()

# 可视化GMM拟合结果
x = np.linspace(140, 210, 1000)
logprob = gmm.score_samples(x.reshape(-1, 1))
responsibilities = gmm.predict_proba(x.reshape(-1, 1))
pdf = np.exp(logprob)
pdf_individual = responsibilities * pdf[:, np.newaxis]

# 将 pdf_individual 打包成元组列表，并按 means 的大小排序
sorted_data = sorted(zip(means, sigmas, weights, pdf_individual.T), key=lambda x: x[0], reverse=True)
# 解压排序后的元组列表
means_sorted, sigmas_sorted, weights_sorted, pdf_individual_sorted = zip(*sorted_data)

# 转换为 NumPy 数组
pdf_individual_sorted = np.array(pdf_individual_sorted).T
means = np.array(means_sorted).flatten()
sigmas = np.array(sigmas_sorted).flatten()
weights = np.array(weights_sorted).flatten()

print("GMM Means:", means)
print("GMM Sigmas:", sigmas)
print("GMM Weights:", weights)
y_m = norm.pdf(x, 176, 8)*0.6
y_f = norm.pdf(x, 164, 6)*0.4

plt.hist(data, bins=30, density=True, alpha=0.6, color='g')
plt.plot(x, pdf, '-k', label='GMM PDF')
plt.plot(x, pdf_individual[:, 0], 'b', label='male PDF')
plt.plot(x, pdf_individual[:, 1], 'r', label='female PDF')
plt.plot(x, y_m, '--b', label='real male PDF')
plt.plot(x, y_f, '--r', label='real female PDF')
plt.legend()
plt.title("GMM Fit")
plt.xlabel("Height (cm)")
plt.ylabel("Density")
plt.show()
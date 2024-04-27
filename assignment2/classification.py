from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from data import make_moons_3d
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Slider
import numpy as np


# 划分数据集
x_test, y_test = make_moons_3d(n_samples=250, noise=0.2)
x_train, y_train = make_moons_3d(n_samples=500, noise=0.2)

# 初始化模型
# 逻辑回归
lr = LogisticRegression()
# 支持向量机
svm_linear = SVC(kernel='linear')
svm_poly = SVC(kernel='poly')
svm_sigmoid = SVC(kernel='sigmoid')
svm_rbf = SVC(kernel='rbf')
# xgboost
xgb = XGBClassifier()

# 训练模型
models = [lr, svm_linear, svm_poly, svm_sigmoid, svm_rbf, xgb]
model_names = ['LogisticRegression', 'SVM (Linear)', 'SVM (poly)', 'SVM (sigmoid)', 'SVM (RBF)', 'XGBoost']
for model, name in zip(models, model_names):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(
        f"{name} performance: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    # 绘制图像
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r' if y_test_e != y_pred_e else 'b' if y_test_e == 0 else 'g' for y_test_e, y_pred_e in zip(y_test, y_pred)]
    colors = np.array(colors)
    scatter = ax.scatter(x_test[:, 0], x_test[:, 1], x_test[:, 2], c=colors, cmap='viridis', marker='o')
    # 设置图例
    error_patch = plt.Line2D([0], [0], marker='o', color='w', label='Error Points', markerfacecolor='r', markersize=10)
    class0_patch = plt.Line2D([0], [0], marker='o', color='w', label='Class 0', markerfacecolor='b', markersize=10)
    class1_patch = plt.Line2D([0], [0], marker='o', color='w', label='Class 1', markerfacecolor='g', markersize=10)
    legend1 = ax.legend(handles=[error_patch, class0_patch, class1_patch], title="Classes")
    legend1.set_bbox_to_anchor((0.1, 1))
    ax.add_artist(legend1)
    Accuracy = plt.Line2D([0], [0], marker='o', color='w', label=f'Accuracy:{accuracy:.4f}', markerfacecolor='w', markersize=10)
    F1 = plt.Line2D([0], [0], marker='o', color='w', label=f'f1_score:{f1:.4f}', markerfacecolor='w', markersize=10)
    legend2= ax.legend(handles=[Accuracy,F1], title="Performance")
    legend2.set_bbox_to_anchor((0.1, 0.2))
    ax.add_artist(legend2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(name)
    plt.show()


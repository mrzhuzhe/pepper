import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

# data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['label'] = iris.target
df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
# data = np.array(df.iloc[:100, [0, 1, -1]])
#  print(df)
"""
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
# plt.show()
"""

data = np.array(df.iloc[:100, [0, 1, -1]])
X, y = data[:,:-1], data[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

class KNN:
    def __init__(self, X_train, y_train, n_neighbors=3, p=2):
        """
        parameter: n_neighbors 临近点个数
        parameter: p 距离度量
        """
        self.n = n_neighbors
        self.p = p
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X):
        # 取出n个点
        knn_list = []
        for i in range(self.n):
            # 计算前三个到当前点的L2范数
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            # 结果放数组
            knn_list.append((dist, self.y_train[i]))        

        for i in range(self.n, len(self.X_train)):
            # 计算三个后面的， 先把最大指向三个中最大的
            max_index = knn_list.index(max(knn_list, key=lambda x: x[0]))
            """
                这个地方肯定是写错了 应该是取最小的才对
                还是应该取最大的不断用更小点换走最大的也就是让 inf(knn_list) 尽可能小
            """
            #min_index = knn_list.index(min(knn_list, key=lambda x: x[0]))
            # 计算当前点到 其他点的范数
            dist = np.linalg.norm(X - self.X_train[i], ord=self.p)
            #if knn_list[min_index][0] > dist:
            if knn_list[max_index][0] > dist:
                print(knn_list[max_index][0], dist)
            #   print(knn_list[min_index][0], dist)
            #   knn_list[min_index] = (dist, self.y_train[i])
                knn_list[max_index] = (dist, self.y_train[i])
        #   print(knn_list)
        # 统计
        knn = [k[-1] for k in knn_list]
        #   print(knn_list)
        count_pairs = Counter(knn)
        #   print(knn_list)
#         max_count = sorted(count_pairs, key=lambda x: x)[-1]
        max_count = sorted(count_pairs.items(), key=lambda x: x[1])[-1][0]
        return max_count

    def score(self, X_test, y_test):
        right_count = 0
        n = 10
        for X, y in zip(X_test, y_test):
            label = self.predict(X)
            if label == y:
                right_count += 1
        return right_count / len(X_test)

clf = KNN(X_train, y_train)

# 这个其实没有训练
res = clf.score(X_test, y_test)
print(res)

test_point = [6.0, 3.0]
print('Test Point: {}'.format(clf.predict(test_point)))

"""
plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
plt.plot(test_point[0], test_point[1], 'bo', label='test_point')
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.legend()
#plt.show()
"""

"""
sklearn.neighbors.KNeighborsClassifier
n_neighbors: 临近点个数
p: 距离度量
algorithm: 近邻算法，可选{'auto', 'ball_tree', 'kd_tree', 'brute'}
weights: 确定近邻的权重

from sklearn.neighbors import KNeighborsClassifier

clf_sk = KNeighborsClassifier()
clf_sk.fit(X_train, y_train)

clf_sk.score(X_test, y_test)

"""
"""
这里的例子有很严重的问题，需要查一下sklern官网
https://scikit-learn.org/stable/modules/tree.html

1. 需要在分类后drop掉当前特征，不然当前特征会被反复使用


"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

# data
def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    #   print(iris.feature_names)
    df['label'] = iris.target
    df.columns = [
        'sepal length', 'sepal width', 'petal length', 'petal width', 'label'
    ]
    data = np.array(df.iloc[:100, [0, 1, -1]])
    #   print(df, data) 只取了前两列和最后一列
    return data[:, :2], data[:, -1]


X, y = create_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#print(X, y)

clf = DecisionTreeClassifier(criterion="entropy", random_state=30)
clf.fit(X_train, y_train,)

print(clf.score(X_test, y_test))
print(clf.feature_importances_)
"""
tree_pic = export_graphviz(clf, out_file="mytree.pdf")
with open('mytree.pdf') as f:
    dot_graph = f.read()

graphviz.Source(dot_graph)
"""
feature_name = [
        'sepal length', 'sepal width'
    ]
tree_pic = export_graphviz(clf, feature_names=feature_name, class_names=["hua", "cao"],  filled=True ,rounded=True)
#tree_pic = export_graphviz(clf, filled=True ,rounded=True)
graph = graphviz.Source(tree_pic)
graph.view()
#print(tree_pic)
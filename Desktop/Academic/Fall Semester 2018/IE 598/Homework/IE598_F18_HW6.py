# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import numpy as np

iris = datasets.load_iris()
X = iris.data
y = iris.target


rndlist=np.arange(1, 11, 1)

dtrain={}
dtest={}

for rnd in rndlist:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=rnd, stratify=y)
    tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
    tree.fit(X_train, y_train)
    train_score = tree.score(X_train, y_train)
    test_score = tree.score(X_test, y_test)
    print(rnd)
    print(train_score)
    print(test_score)
    dtrain[rnd] = train_score
    dtest[rnd] = test_score

d = float(sum(dtrain.values())) / len(dtrain)
print(d)
h = float(sum(dtest.values())) / len(dtest)
print(h)


from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)
tree = DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1)
scores_set1 = cross_val_score(tree, X, y=y, cv=10)




print("My name is {Audrey Ayo}")
print("My NetID is: {ayo2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

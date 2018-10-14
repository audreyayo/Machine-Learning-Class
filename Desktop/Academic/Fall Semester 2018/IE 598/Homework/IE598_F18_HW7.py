# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:04:17 2018

@author: Audrey
"""


import pandas as pd

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols',
                   'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue',
                   'OD280/OD315 of diluted wines', 'Proline']

df_wine.head()

from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

cv_scores = []
ne_range = [1, 5, 10, 50, 100,500,1000,5000]
for n_e in ne_range:
    time_start = time.clock()
    #run your code
    rf = RandomForestClassifier(n_estimators = n_e, random_state=0, n_jobs=-1)
    score = cross_val_score(rf, X = X_train, y = y_train, cv=10)
    cv_scores.append(score)
    time_elapsed = (time.clock() - time_start)
    print(time_elapsed)
    print('CV accuracy score: %s' % score)
    print('CV accuracy: %.3f +/- %.3f' % (np.mean(score), np.std(score))) 
    

    
feat_labels = df_wine.columns[1:]
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train) 
importances = forest.feature_importances_ 
indices = np.argsort(importances)[::-1] 
for f in range(X_train.shape[1]): 
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[f], importances[indices[f]]))

import matplotlib.pyplot as plt
    
plt.title('Feature Importances') 
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')  
plt.xticks(range(X_train.shape[1]), feat_labels, rotation=90) 
plt.xlim([-1, X_train.shape[1]]) 
plt.tight_layout() 
plt.show()

print("My name is {Audrey Ayo}")
print("My NetID is: {ayo2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

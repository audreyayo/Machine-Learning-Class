# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 09:53:52 2018
@author: Audrey
"""




#Loading the Wine dataset
import pandas as pd
df_wine = pd.read_csv(r"C:\Users\Audrey\Desktop\Academic\Fall Semester 2018\IE 598\Homework\Homework 5\wine.csv", header=None)

df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash','Alkalinity of ash', 'Magnesium', 'Total phenols','Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins','Color intensity', 'Hue','OD280/OD315 of diluted wines', 'Proline']
#
print(df_wine.head())
print(df_wine.info())
print(df_wine.groupby(['Class label']).describe())

boxplot = df_wine.boxplot(column=['Proline'], by='Class label')
plt.suptitle("")
plt.title("")

#Process the Wine data into seperate training and test sets using 80% training data and 20% test data
from sklearn.cross_validation import train_test_split 
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,stratify=y, random_state=42)

#Standardize the data to unit variance
from sklearn.preprocessing import StandardScaler
sc = StandardScaler() 
X_train_std = sc.fit_transform(X_train)  
X_test_std = sc.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train_std,y_train)

y_test_pred = lr.predict(X_test_std)
y_train_pred = lr.predict(X_train_std)

from sklearn import metrics
print( metrics.accuracy_score(y_train, y_train_pred) )
print( metrics.accuracy_score(y_test, y_test_pred) )

print( metrics.classification_report(y_test, y_test_pred))
print( metrics.classification_report(y_train, y_train_pred))

print( metrics.confusion_matrix(y_test, y_test_pred))
print( metrics.confusion_matrix(y_train, y_train_pred))


from sklearn.svm import SVC

svc.fit(X_train_std, y_train)

y_test_pred = svc.predict(X_test_std)
y_train_pred = svc.predict(X_train_std)

print( metrics.accuracy_score(y_train, y_train_pred) )
print( metrics.accuracy_score(y_test, y_test_pred) )

print("SVM Classifier Report in test set:")
print(metrics.classification_report(y_test, y_test_pred))
print("SVM Classifier Report in train set:")
print(metrics.classification_report(y_train, y_train_pred))

print( metrics.confusion_matrix(y_test, y_test_pred))
print( metrics.confusion_matrix(y_train, y_train_pred))

#The eigenvectors of the covariance matrix represent the principal components
import numpy as np
cov_mat = np.cov(X_train_std.T) #computing the covariance matrix
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat) #computing eigenvalues and eigenvectors  
print('\nEigenvalues \n%s' % eigen_vals) #print eigenvalues

tot = sum(eigen_vals) 
var_exp = [(i / tot) for i in            
           sorted(eigen_vals, reverse=True)] 
cum_var_exp = np.cumsum(var_exp)
import matplotlib.pyplot as plt 
plt.bar(range(1,14), var_exp, alpha=0.5, align='center',label='individual explained variance')
plt.step(range(1,14), cum_var_exp, where='mid',label='cumulative explained variance')
plt.ylabel('Explained variance ratio') 
plt.xlabel('Principal components') 
plt.legend(loc='best') 
plt.show() 

from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

svc.fit(X_train_pca, y_train)

y_test_pred = svc.predict(X_test_pca)
y_train_pred = svc.predict(X_train_pca,)

print( metrics.accuracy_score(y_train, y_train_pred) )
print( metrics.accuracy_score(y_test, y_test_pred) )

print( metrics.classification_report(y_test, y_test_pred))
print( metrics.classification_report(y_train, y_train_pred))

print( metrics.confusion_matrix(y_test, y_test_pred))
print( metrics.confusion_matrix(y_train, y_train_pred))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)
X_test_lda = lda.transform(X_test_std)

#lr = LogisticRegression()
svc.fit(X_train_lda, y_train)

y_test_pred = svc.predict(X_test_lda)
y_train_pred = svc.predict(X_train_lda,)

print( metrics.accuracy_score(y_train, y_train_pred) )
print( metrics.accuracy_score(y_test, y_test_pred) )

print( metrics.classification_report(y_test, y_test_pred))
print( metrics.classification_report(y_train, y_train_pred))

print( metrics.confusion_matrix(y_test, y_test_pred))
print( metrics.confusion_matrix(y_train, y_train_pred))


from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score

for g in [0.1, 1, 10]:
    kpca = KernelPCA(kernel="rbf", gamma=g)
    X_train_kpca = kpca.fit_transform(X_train_std)
    X_test_kpca = kpca.transform(X_test_std)
    
    svc=LogisticRegression()
    svc.fit(X_train_std,y_train)
    #svc = SVC()
    #svc.fit(X_train_kpca, y_train)

    y_test_predict = svc.predict(X_test_kpca)
    y_train_predict = svc.predict(X_train_kpca)
    print("Gamma: ", g)
    print(accuracy_score(y_test, y_test_predict))
    print(accuracy_score(y_train, y_train_predict))
    print( metrics.classification_report(y_test, y_test_predict))
    print( metrics.classification_report(y_train, y_train_predict))

    print( metrics.confusion_matrix(y_test, y_test_predict))
    print( metrics.confusion_matrix(y_train, y_train_predict))



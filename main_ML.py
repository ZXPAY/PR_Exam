'''
Compile environment : anaconda python 3.6
Using Manny method and change the library to enhance precision,
    SVM (Support Vector Machine)
    Tree (Decision Tree)
    Bayes (Naive Bayes Decision)
    KNN (K Nearest Neighbor)
    XGB (XG Boost)
    Stacking (Stacking)
    Voting (Voting, 得票數最高決定預測值)
    Bagging (Bagging)
    RF (Random Forest)
    Adaboost (Adaptive Boost)
'''

from Danny_ML_CLF import Danny_ML_CLF
import numpy as np
from sklearn.preprocessing import StandardScaler

# Claim the parameters
features = 2
DEBUG = False

X = np.array([[3.98,1,2.5],
              [1.2,1.78,3.4],
              [8.9,8.8,7.9],
              [9.8,8.7,9.01],
              [1.22,0.6,2.1],
              [7.8,9.5,8.45]],dtype=np.float32)
y = np.array([1,1,2,2,1,2],dtype=np.float32)
test_X=np.array([[5,6,7],[1,1,1],[0.8,9,3.5]],dtype=np.float32)

if DEBUG:
    print('shape X:', X.shape)
    print('shape y:', y.shape)
    print('X:\n', X)
    print('y:\n', y)

sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)
test_X = sc.transform(test_X)

# Code begin
myclf = Danny_ML_CLF()

myclf.Fit_value(X, y)
myclf.Train()
myclf.Report(X, y, [0, 1], show_cm=False)
myclf.Score()

print(myclf.SVM_predict(test_X))
print(myclf.Tree_predict(test_X))
print(myclf.Bayes_predict(test_X))
print(myclf.KNN_predict(test_X))
print(myclf.XGB_prediction(test_X))
print(myclf.Stacking_prediction(test_X))
print(myclf.Voting_prediction(test_X))
print(myclf.Bagging_prediction(test_X))
print(myclf.RF_prediction(test_X))
print(myclf.Adaboost_prediction(test_X))


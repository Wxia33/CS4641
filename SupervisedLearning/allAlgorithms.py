import time
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

decisionTree = tree.DecisionTreeClassifier()
svm_class = svm.SVC(kernel = 'linear', coef0 = coef)
knnClass = KNeighborsClassifier(algorithm = algm[i])
nn_mlp = MLPClassifier(solver='lbfgs', alpha= alph)
adaBoost = AdaBoostClassifier(n_estimators = n_est)

data = './data/MNIST/'

# data obtained from https://www.kaggle.com/oddrationale/mnist-in-csv#mnist_test.csv
train = data + 'mnist_train.csv'

svm_train = pd.read_csv(train)

trainLabel = svm_train.label
trained = svm_train.drop('label',1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(trained, trainLabel, test_size = 30000, random_state = 42)

import pandas as pd
from sklearn import svm

# Code from https://scikit-learn.org/stable/modules/svm.html

data = './data/MNIST/'

# data obtained from https://www.kaggle.com/oddrationale/mnist-in-csv#mnist_test.csv
testMnist = data + 'mnist_test.csv'
trainMnist = data + 'mnist_train.csv'

mnist_train = pd.read_csv(trainMnist)
mnist_test = pd.read_csv(testMnist)

trainLabel = mnist_train.label
trained = mnist_train.drop('label',1)

svm_class = svm.SVC(gamma='scale')
svm_class.fit(trained, trainLabel)

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

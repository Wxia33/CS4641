import time
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

# Adaboost classifier

#iris = load_iris()

#scores = cross_val_score(clf, iris.data, iris.target, cv=5)
#scores.mean()

data = './data/MNIST/'

# data obtained from https://www.kaggle.com/oddrationale/mnist-in-csv#mnist_test.csv
testMnist = data + 'mnist_test.csv'
trainMnist = data + 'mnist_train.csv'

mnist_train = pd.read_csv(trainMnist)
mnist_test = pd.read_csv(testMnist)

trainLabel = mnist_train.label
trained = mnist_train.drop('label',1)

mnist_label = mnist_test.label
mnist_test = mnist_test.drop('label',1)

train_size_list = []
time_train_list = []
predict_train_list = []
accuracy_list = []
num_nodes_list = []

for i in range(0,19):
    t_size = 1 + 20 * i
    adaBoost = AdaBoostClassifier(n_estimators = t_size)

    print '----------------------------------------------------------------'
    print 'TESTING SIZE: ', t_size

    startt = time.time()
    adaBoost.fit(trained, trainLabel)
    endt = time.time()

    print 'Training Size', len(trained)
    print 'Test Size', len(mnist_test)
    print 'Time to train Adaboost', endt - startt

    start = time.time()
    adaBoostPredResult = adaBoost.predict(mnist_test)
    end = time.time()

    print 'Time to Predict with Adaboost', end - start

    train_size_list.append(len(trained))
    time_train_list.append(endt - startt)
    predict_train_list.append(end - start)

    correct = 0
    total = 0
    for i,j in zip(adaBoostPredResult,mnist_label):
        if i == j:
            correct += 1.0
        total += 1.0

    print 'Accuracy', correct / total

    accuracy_list.append(correct / total)

finalReport = pd.DataFrame(data = train_size_list, columns=['Training Size'])
finalReport['Time to Train Adaboost'] = time_train_list
finalReport['Time to Predict with Adaboost'] = predict_train_list
finalReport['Accuracy'] = accuracy_list

finalReport.to_csv('./adaboostReport_MNIST.csv',index=False)

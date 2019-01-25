import pandas as pd
import time
from sklearn import model_selection
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

mnist_label = mnist_test.label
mnist_test = mnist_test.drop('label',1)

svm_class = svm.SVC(gamma='scale')

train_size_list = []
time_train_list = []
predict_train_list = []
accuracy_list = []
num_nodes_list = []

for i in range(0,19):
    t_size = 0.9 - i * 0.05
    print '----------------------------------------------------------------'
    print 'TESTING SIZE: ', t_size


    X_train, X_test, y_train, y_test = model_selection.train_test_split(trained, trainLabel, test_size = t_size, random_state = 42)

    startt = time.time()
    svm_class.fit(trained, trainLabel)
    endt = time.time()

    print 'Training Size', len(X_train)
    print 'Test Size', len(mnist_test)
    print 'Time to train SVM', endt - startt

    start = time.time()
    svmPredictResult = svm_class.predict(mnist_test, mnist_label)
    end = time.time()

    print 'Time to Predict with SVM', end - start

    train_size_list.append(len(X_train))
    time_train_list.append(endt - startt)
    predict_train_list.append(end - start)

    correct = 0
    total = 0
    for i,j in zip(svmPredictResult,mnist_label):
        if i == j:
            correct += 1.0
        total += 1.0

    print 'Accuracy', correct / total

    accuracy_list.append(correct / total)

finalReport = pd.DataFrame(data = train_size_list, columns=['Training Size'])
finalReport['Time to Train SVM'] = time_train_list
finalReport['Time to Predict with SVM'] = predict_train_list
finalReport['Accuracy'] = accuracy_list

finalReport.to_csv('./svmReport_MNIST.csv',index=False)

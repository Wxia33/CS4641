import sklearn
import time
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import model_selection

data = './data/MNIST/'

# data obtained from https://www.kaggle.com/oddrationale/mnist-in-csv#mnist_test.csv
test = data + 'mnist_test.csv'
train = data + 'mnist_train.csv'

decision_train = pd.read_csv(train)
decision_test = pd.read_csv(test)

decisionTree = tree.DecisionTreeClassifier()

trainLabel = decision_train.label
trained = decision_train.drop('label',1)

decision_label = decision_test.label
decision_test = decision_test.drop('label',1)

#trainLabel = np.array(trainLabel)
#trained = np.array(trained)
#decision_label = np.array(decision_label)
#decision_test = np.array(decision_test)

train_size_list = []
time_train_list = []
predict_train_list = []
accuracy_list = []
num_nodes_list = []

for i in range(0,20):
    t_size = 0.9 - i * 0.05
    print '----------------------------------------------------------------'
    print 'TESTING SIZE: ', t_size
    X_train, X_test, y_train, y_test = model_selection.train_test_split(trained, trainLabel, test_size = t_size, random_state = 42)

    # Timing Decision Tree Fit
    start = time.time()
    dtree = decisionTree.fit(X_train,y_train)
    end = time.time()

    print 'Training Size', len(X_train)
    print 'Test Size', len(decision_test)
    print 'Time to train Decision Tree', end - start

    train_size_list.append(len(X_train))
    time_train_list.append(end - start)

    startPred = time.time()
    predictResult = decisionTree.predict(decision_test)
    endPred = time.time()
    print 'Time to predict with Decision Tree', endPred - startPred
    predict_train_list.append(endPred - startPred)

    resultsPredict = pd.DataFrame(data=predictResult, columns=['label'])
    resultsPredict['ImageId'] = list(range(1,len(predictResult)+1))

    #resultsPredict.to_csv('./mnist_dt_t-size0.1.csv',index = False)

    correct = 0
    total = 0
    for i,j in zip(resultsPredict['label'],decision_label):
        if i == j:
            correct += 1.0
        total += 1.0

    print 'Accuracy', correct / total
    print 'Number of Nodes', dtree.tree_.node_count

    accuracy_list.append(correct / total)
    num_nodes_list.append(dtree.tree_.node_count)

finalReport = pd.DataFrame(data = train_size_list, columns=['Training Size'])
finalReport['Time to Train DT'] = time_train_list
finalReport['Time to Predict with DT'] = predict_train_list
finalReport['Accuracy'] = accuracy_list
finalReport['Number of Nodes'] = num_nodes_list

finalReport.to_csv('./dtReport_MNIST.csv',index=False)

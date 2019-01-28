import sklearn
import time
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn import model_selection
from sklearn.model_selection import cross_validate

data = './data/MNIST/'

# data obtained from https://www.kaggle.com/oddrationale/mnist-in-csv#mnist_test.csv
test = data + 'mnist_test.csv'
train = data + 'mnist_train.csv'

decision_train = pd.read_csv(train)
decision_test = pd.read_csv(test)

decisionTree = tree.DecisionTreeClassifier()

trainLabel = decision_train.label
trained = decision_train.drop('label',1)
X_train, X_test, y_train, y_test = model_selection.train_test_split(trained, trainLabel, test_size = 30000, random_state = 42)
trainLabel = y_train
trained = X_train

decision_label = decision_test.label
decision_test = decision_test.drop('label',1)

nLeaf_list = []
fold_size_list = []
time_train_list = []
predict_train_list = []
num_nodes_list = []
train_score_list = []
val_score_list = []

for i in range(0,8):
    print '----------------------------------------------------------------'
    print 'FOLD COUNT: ', 6

    n_leaf = 100 + 300 * i
    decisionTree = tree.DecisionTreeClassifier(max_leaf_nodes = n_leaf)

    print 'Max Leaf: ', n_leaf

    cvEst = cross_validate(decisionTree, trained, trainLabel, cv = 6, return_train_score = True)

    print 'Time to train Decision Tree', cvEst['fit_time']

    print 'Time to Predict with Decision Tree', cvEst['score_time']

    print 'Training Score', cvEst['train_score']
    print 'Testing Score', cvEst['test_score']

    nLeaf_list.append(n_leaf)
    fold_size_list.append(6)
    time_train_list.append(np.average(cvEst['fit_time']))
    predict_train_list.append(np.average(cvEst['score_time']))
    train_score_list.append(np.average(cvEst['train_score']))
    val_score_list.append(np.average(cvEst['test_score']))

outDat = {
    'Max Leaf': nLeaf_list,
    '# of Folds': fold_size_list,
    'Time to Train Decision Tree': time_train_list,
    'Time to Predict with Decision Tree': predict_train_list,
    'Training Accuracy': train_score_list,
    'Validation Accuracy': val_score_list
}

finalReport = pd.DataFrame(outDat, columns = [
                'Max Leaf',
                '# of Folds',
                'Time to Train Decision Tree',
                'Time to Predict with Decision Tree',
                'Training Accuracy',
                'Validation Accuracy'])

finalReport.to_csv('./dtReport_MaxLeaf_MNIST.csv',index=False)

import sklearn
import numpy as np
import pandas as pd
from sklearn import tree

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

decisionTree.fit(trained,trainLabel)

predictResult = decisionTree.predict(decision_test)

resultsPredict = pd.DataFrame(data=predictResult, columns=['label'])
resultsPredict['ImageId'] = list(range(1,len(decision_test)+1))

resultsPredict.to_csv('mnist_dt.csv',index = False)

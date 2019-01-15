import sklearn
import numpy as np
import pandas as pd
from sklearn import tree

data = './data/MNIST/'

test = data + 'mnist_test.csv'
train = data + 'mnist_train.csv'

decision_train = pd.read_csv(train)
decision_test = pd.read_csv(test)

decisionTree = tree.DecisionTreeClassifier()

trainLabel = train.label
train = train.drop('label',1)

decisionTree.fit(train,trainLabel)

predictResult = decisionTree.predict(test)

resultsPredict = pd.DataFrame(data=dt, columns=['label'])
resultsPredict['ImageId'] = list(range(1,len(test)+1))

resultsPredict.to_csv('mnist_dt.csv',index = False)

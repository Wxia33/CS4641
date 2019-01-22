import numpy as np
import pandas as pd

data = '../data/MNIST/'

# data obtained from https://www.kaggle.com/oddrationale/mnist-in-csv#mnist_test.csv
test = data + 'mnist_test.csv'
train = data + 'mnist_train.csv'

x = 0

'''
Analyze decision tree result on MNIST dataset
'''
if x == 0:
    #decision_train = pd.read_csv(train)
    decision_test = pd.read_csv(test)

    #trainLabel = decision_train.label
    #trained = decision_train.drop('label',1)

    decision_label = decision_test.label
    decision_test = decision_test.drop('label',1)

    mnist_decision_res = pd.read_csv('../mnist_dt.csv')


    correct = 0
    total = 0
    for i,j in zip(mnist_decision_res['label'],decision_label):
        if i == j:
            correct += 1.0
        total += 1.0

    print correct / total # 88.02% efficiency [12:25am, 1/22/2019]

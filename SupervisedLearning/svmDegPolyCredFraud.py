import pandas as pd
import time
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn import svm

# Code from https://scikit-learn.org/stable/modules/svm.html

data = './data/'

#Data obtained from https://www.kaggle.com/mlg-ulb/creditcardfraud
credCard = data + 'creditCardTrain.csv'
credCard = pd.read_csv(credCard)

trainLabel = credCard.Class
trained = credCard.drop('Class',1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(trained, trainLabel, test_size = 150000, random_state = 42)
trainLabel = y_train
trained = X_train

mnist_label = mnist_test.label
mnist_test = mnist_test.drop('label',1)

degree_list = []
fold_size_list = []
time_train_list = []
predict_train_list = []
num_nodes_list = []
train_score_list = []
val_score_list = []

for i in range(1,9):
    print '----------------------------------------------------------------'
    print 'FOLD COUNT: ', 6

    deg = i
    svm_class = svm.SVC(kernel = 'poly', degree = deg)

    print 'Degree of Polynomial Kernel Function: ', deg

    cvEst = cross_validate(svm_class, trained, trainLabel, cv = 6, return_train_score = True)

    print 'Time to train SVM', cvEst['fit_time']

    print 'Time to Predict with SVM', cvEst['score_time']

    print 'Training Score', cvEst['train_score']
    print 'Testing Score', cvEst['test_score']

    degree_list.append(deg)
    fold_size_list.append(6)
    time_train_list.append(np.average(cvEst['fit_time']))
    predict_train_list.append(np.average(cvEst['score_time']))
    train_score_list.append(np.average(cvEst['train_score']))
    val_score_list.append(np.average(cvEst['test_score']))

outDat = {
    'Degree of Polynomial Kernel Function': degree_list,
    '# of Folds': fold_size_list,
    'Time to Train SVM': time_train_list,
    'Time to Predict with SVM': predict_train_list,
    'Training Accuracy': train_score_list,
    'Validation Accuracy': val_score_list
}

finalReport = pd.DataFrame(outDat, columns = [
                'Degree of Polynomial Kernel Function',
                '# of Folds',
                'Time to Train SVM',
                'Time to Predict with SVM',
                'Training Accuracy',
                'Validation Accuracy'])

finalReport.to_csv('./svmReport_DegreePoly_MNIST.csv',index=False)

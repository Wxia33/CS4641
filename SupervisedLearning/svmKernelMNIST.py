import time
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler

data = './data/MNIST/'

# data obtained from https://www.kaggle.com/oddrationale/mnist-in-csv#mnist_test.csv
train = data + 'mnist_train.csv'

svm_train = pd.read_csv(train)

trainLabel = svm_train.label
trained = svm_train.drop('label',1)

trained = MinMaxScaler().fit_transform(trained)

kern_list = []
fold_size_list = []
time_train_list = []
predict_train_list = []
num_nodes_list = []
train_score_list = []
val_score_list = []

for i in range(0,4):
    print '----------------------------------------------------------------'
    print 'FOLD COUNT: ', 5

    kern = ['linear','poly','rbf','sigmoid']

    svm_class = svm.SVC(kernel = kern[i], cache_size = 1000)

    print 'Kernel Function: ', kern[i]

    cvEst = cross_validate(svm_class, trained, trainLabel, cv = 5, return_train_score = True)

    print 'Time to train SVM', cvEst['fit_time']

    print 'Time to Predict with SVM', cvEst['score_time']

    print 'Training Score', cvEst['train_score']
    print 'Testing Score', cvEst['test_score']

    kern_list.append(coef)
    fold_size_list.append(6)
    time_train_list.append(np.average(cvEst['fit_time']))
    predict_train_list.append(np.average(cvEst['score_time']))
    train_score_list.append(np.average(cvEst['train_score']))
    val_score_list.append(np.average(cvEst['test_score']))

outDat = {
    'Kernel Function': kern_list,
    '# of Folds': fold_size_list,
    'Time to Train SVM': time_train_list,
    'Time to Predict with SVM': predict_train_list,
    'Training Accuracy': train_score_list,
    'Validation Accuracy': val_score_list
}

finalReport = pd.DataFrame(outDat, columns = [
                'Kernel Function',
                '# of Folds',
                'Time to Train SVM',
                'Time to Predict with SVM',
                'Training Accuracy',
                'Validation Accuracy'])

finalReport.to_csv('./reports/wine/svmReport_Kernel_MNIST.csv',index=False)

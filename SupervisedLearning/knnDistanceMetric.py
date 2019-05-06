import time
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate

data = './data/MNIST/'

# data obtained from https://www.kaggle.com/oddrationale/mnist-in-csv#mnist_test.csv
train = data + 'mnist_train.csv'

svm_train = pd.read_csv(train)

trainLabel = svm_train.label
trained = svm_train.drop('label',1)

alg_list = []
fold_size_list = []
time_train_list = []
predict_train_list = []
num_nodes_list = []
train_score_list = []
val_score_list = []

for i in range(0,7):
    print '----------------------------------------------------------------'
    print 'FOLD COUNT: ', 6

    algm = ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    knnClass = KNeighborsClassifier(metric = algm[i], n_jobs = -1)

    print 'Distance Metric: ', algm[i]

    cvEst = cross_validate(knnClass, trained, trainLabel, cv = 6, return_train_score = True)

    print 'Time to train KNN', cvEst['fit_time']

    print 'Time to Predict with KNN', cvEst['score_time']

    print 'Training Score', cvEst['train_score']
    print 'Testing Score', cvEst['test_score']

    alg_list.append(algm[i])
    fold_size_list.append(6)
    time_train_list.append(np.average(cvEst['fit_time']))
    predict_train_list.append(np.average(cvEst['score_time']))
    train_score_list.append(np.average(cvEst['train_score']))
    val_score_list.append(np.average(cvEst['test_score']))

outDat = {
    'Distance Metric': alg_list,
    '# of Folds': fold_size_list,
    'Time to Train KNN': time_train_list,
    'Time to Predict with KNN': predict_train_list,
    'Training Accuracy': train_score_list,
    'Validation Accuracy': val_score_list
}

finalReport = pd.DataFrame(outDat, columns = [
                'Distance Metric',
                '# of Folds',
                'Time to Train KNN',
                'Time to Predict with KNN',
                'Training Accuracy',
                'Validation Accuracy'])

finalReport.to_csv('./reports/knnReport_DistMetric_MNIST.csv',index=False)

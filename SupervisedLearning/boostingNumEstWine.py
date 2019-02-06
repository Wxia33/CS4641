import time
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_validate

data = './data/'

# data obtained from https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
train = data + 'wineQualityTrain.csv'

boost_train = pd.read_csv(train)

trainLabel = boost_train['11']
trained = boost_train.drop('11',1)

est_list = []
fold_size_list = []
time_train_list = []
predict_train_list = []
num_nodes_list = []
train_score_list = []
val_score_list = []

for i in range(0,9):
    print '----------------------------------------------------------------'
    print 'FOLD COUNT: ', 6

    n_est = 1 + 10 * i
    adaBoost = AdaBoostClassifier(n_estimators = n_est)

    print 'Coefficient of Polynomial Kernel Function: ', n_est

    cvEst = cross_validate(adaBoost, trained, trainLabel, cv = 6, return_train_score = True)

    print 'Time to train Adaboost', cvEst['fit_time']

    print 'Time to Predict with Adaboost', cvEst['score_time']

    print 'Training Score', cvEst['train_score']
    print 'Testing Score', cvEst['test_score']

    est_list.append(n_est)
    fold_size_list.append(6)
    time_train_list.append(np.average(cvEst['fit_time']))
    predict_train_list.append(np.average(cvEst['score_time']))
    train_score_list.append(np.average(cvEst['train_score']))
    val_score_list.append(np.average(cvEst['test_score']))

outDat = {
    'Number of Estimators': est_list,
    '# of Folds': fold_size_list,
    'Time to Train Boost': time_train_list,
    'Time to Predict with Boost': predict_train_list,
    'Training Accuracy': train_score_list,
    'Validation Accuracy': val_score_list
}

finalReport = pd.DataFrame(outDat, columns = [
                'Number of Estimators',
                '# of Folds',
                'Time to Train Boost',
                'Time to Predict with Boost',
                'Training Accuracy',
                'Validation Accuracy'])

finalReport.to_csv('./reports/wine/boostingReport_NEstim_Wine.csv',index=False)

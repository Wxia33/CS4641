import time
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier

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

alph_list = []
fold_size_list = []
time_train_list = []
predict_train_list = []
num_nodes_list = []
train_score_list = []
val_score_list = []

for i in range(0,7):
    print '----------------------------------------------------------------'
    print 'FOLD COUNT: ', 5

    alph = 1 * 10 ** - (8 - 1 * i)
    nn_mlp = MLPClassifier(solver='lbfgs', alpha= alph)

    print 'Alpha: ', alph

    cvEst = cross_validate(nn_mlp, trained, trainLabel, cv = 5, return_train_score = True)

    print 'Time to train NN', cvEst['fit_time']

    print 'Time to Predict with NN', cvEst['score_time']

    print 'Training Score', cvEst['train_score']
    print 'Testing Score', cvEst['test_score']

    alph_list.append(alph)
    fold_size_list.append(6)
    time_train_list.append(np.average(cvEst['fit_time']))
    predict_train_list.append(np.average(cvEst['score_time']))
    train_score_list.append(np.average(cvEst['train_score']))
    val_score_list.append(np.average(cvEst['test_score']))

outDat = {
    'Alpha': alph_list,
    '# of Folds': fold_size_list,
    'Time to Train NN': time_train_list,
    'Time to Predict with NN': predict_train_list,
    'Training Accuracy': train_score_list,
    'Validation Accuracy': val_score_list
}

finalReport = pd.DataFrame(outDat, columns = [
                'Alpha',
                '# of Folds',
                'Time to Train NN',
                'Time to Predict with NN',
                'Training Accuracy',
                'Validation Accuracy'])

finalReport.to_csv('./reports/nnReport_alph_MNIST2.csv',index=False)

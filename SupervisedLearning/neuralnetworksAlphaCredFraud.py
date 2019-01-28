import time
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate

data = './data/'

#Data obtained from https://www.kaggle.com/mlg-ulb/creditcardfraud
credCard = data + 'creditCardTrain.csv'
credCard = pd.read_csv(credCard)

trainLabel = credCard.Class
trained = credCard.drop('Class',1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(trained, trainLabel, test_size = 150000, random_state = 42)
trainLabel = y_train
trained = X_train

alph_list = []
fold_size_list = []
time_train_list = []
predict_train_list = []
num_nodes_list = []
train_score_list = []
val_score_list = []

for i in range(0,9):
    print '----------------------------------------------------------------'
    print 'FOLD COUNT: ', 6

    alph = 1 * 10 ** - (5 + 0.1 * i)
    nn_mlp = MLPClassifier(solver='lbfgs', alpha= alph)

    print 'Alpha: ', alph

    cvEst = cross_validate(nn_mlp, trained, trainLabel, cv = 6, return_train_score = True)

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

finalReport.to_csv('./nnReport_alph_MNIST.csv',index=False)

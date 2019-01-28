import time
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn import model_selection
from sklearn.model_selection import cross_validate

# Adaboost classifier

#iris = load_iris()

#scores = cross_val_score(clf, iris.data, iris.target, cv=5)
#scores.mean()

data = './data/MNIST/'

# data obtained from https://www.kaggle.com/oddrationale/mnist-in-csv#mnist_test.csv
testMnist = data + 'mnist_test.csv'
trainMnist = data + 'mnist_train.csv'

mnist_train = pd.read_csv(trainMnist)
mnist_test = pd.read_csv(testMnist)

trainLabel = mnist_train.label
trained = mnist_train.drop('label',1)
X_train, X_test, y_train, y_test = model_selection.train_test_split(trained, trainLabel, test_size = 30000, random_state = 42)
trainLabel = y_train
trained = X_train

mnist_label = mnist_test.label
mnist_test = mnist_test.drop('label',1)

lRate_list = []
fold_size_list = []
time_train_list = []
predict_train_list = []
num_nodes_list = []
train_score_list = []
val_score_list = []

for i in range(0,9):
    print '----------------------------------------------------------------'
    print 'FOLD COUNT: ', 6

    lRate = 0.6 + 0.1 * i
    adaBoost = AdaBoostClassifier(learning_rate = lRate)

    print 'Coefficient of Polynomial Kernel Function: ', lRate

    cvEst = cross_validate(adaBoost, trained, trainLabel, cv = 6, return_train_score = True)

    print 'Time to train Adaboost', cvEst['fit_time']

    print 'Time to Predict with Adaboost', cvEst['score_time']

    print 'Training Score', cvEst['train_score']
    print 'Testing Score', cvEst['test_score']

    lRate_list.append(lRate)
    fold_size_list.append(6)
    time_train_list.append(np.average(cvEst['fit_time']))
    predict_train_list.append(np.average(cvEst['score_time']))
    train_score_list.append(np.average(cvEst['train_score']))
    val_score_list.append(np.average(cvEst['test_score']))

outDat = {
    'Learning Rate': lRate_list,
    '# of Folds': fold_size_list,
    'Time to Train Boost': time_train_list,
    'Time to Predict with Boost': predict_train_list,
    'Training Accuracy': train_score_list,
    'Validation Accuracy': val_score_list
}

finalReport = pd.DataFrame(outDat, columns = [
                'Learning Rate',
                '# of Folds',
                'Time to Train Boost',
                'Time to Predict with Boost',
                'Training Accuracy',
                'Validation Accuracy'])

finalReport.to_csv('./boostingReport_lRate_MNIST.csv',index=False)

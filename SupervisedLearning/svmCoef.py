import pandas as pd
import time
from sklearn import model_selection
from sklearn import svm

# Code from https://scikit-learn.org/stable/modules/svm.html

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

coef_list = []
fold_size_list = []
time_train_list = []
predict_train_list = []
num_nodes_list = []
train_score_list = []
val_score_list = []

for i in range(0,10):
    print '----------------------------------------------------------------'
    print 'FOLD COUNT: ', 6

    coef = -4 + 1 * i
    svm_class = svm.SVC(kernel = 'poly', coef0 = coef)

    print 'Coefficient of Polynomial Kernel Function: ', coef

    cvEst = cross_validate(svm_class, trained, trainLabel, cv = 6, return_train_score = True)

    print 'Time to train SVM', cvEst['fit_time']

    print 'Time to Predict with SVM', cvEst['score_time']

    print 'Training Score', cvEst['train_score']
    print 'Testing Score', cvEst['test_score']

    coef_list.append(coef)
    fold_size_list.append(6)
    time_train_list.append(np.average(cvEst['fit_time']))
    predict_train_list.append(np.average(cvEst['score_time']))
    train_score_list.append(np.average(cvEst['train_score']))
    val_score_list.append(np.average(cvEst['test_score']))

outDat = {
    'Coefficient of Polynomial Kernel Function': coef_list,
    '# of Folds': fold_size_list,
    'Time to Train SVM': time_train_list,
    'Time to Predict with SVM': predict_train_list,
    'Training Accuracy': train_score_list,
    'Validation Accuracy': val_score_list
}

finalReport = pd.DataFrame(outDat, columns = [
                'Coefficient of Polynomial Kernel Function',
                '# of Folds',
                'Time to Train SVM',
                'Time to Predict with SVM',
                'Training Accuracy',
                'Validation Accuracy'])

finalReport.to_csv('./svmReport_DegreePoly_MNIST.csv',index=False)

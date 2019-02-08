import pandas as pd
import time
from sklearn import model_selection
from sklearn import svm

data = './data/'

# data obtained from https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
train = data + 'wineQualityTrain.csv'

svm_train = pd.read_csv(train)

trainLabel = svm_train['11']
trained = svm_train.drop('11',1)

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

finalReport.to_csv('./reports/wine/svmReport_DegreePoly_Wine.csv',index=False)

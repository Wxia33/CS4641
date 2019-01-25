import time
import pandas as pd
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate

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

fold_size_list = []
time_train_list = []
predict_train_list = []
num_nodes_list = []
train_score_list = []
val_score_list = []

for i in range(0,10):
    print '----------------------------------------------------------------'
    print 'TESTING SIZE: ', 0.3

    n_nb = 1 + 3 * i
    knnClass = KNeighborsClassifier(n_neighbors = n_nb)

    print 'Number of Neighbors: ', n_nb

    cvEst = cross_validate(knnClass, trained, trainLabel, cv = 6, return_train_score = True)

    print 'Time to train KNN', cvEst['fit_time']

    print 'Time to Predict with KNN', cvEst['score_time']

    print 'Training Score', cvEst['train_score']
    print 'Testing Score', cvEst['test_score']

    fold_size_list.append(6)
    time_train_list.append(cvEst['fit_time'])
    predict_train_list.append(cvEst['score_time'])
    train_score_list.append(np.average(cvEst['train_score']))
    val_score_list.append(np.average(cvEst['test_score']))

    '''
    correct = 0
    total = 0
    for i,j in zip(knnPredictResult,mnist_label):
        if i == j:
            correct += 1.0
        total += 1.0

    print 'Accuracy', correct / total

    accuracy_list.append(correct / total)
    '''

finalReport = pd.DataFrame(data = train_size_list, columns=['Training Size'])
finalReport['Time to Train KNN'] = time_train_list
finalReport['Time to Predict with KNN'] = predict_train_list
finalReport['Accuracy'] = accuracy_list

finalReport.to_csv('./knnReport_MNIST.csv',index=False)

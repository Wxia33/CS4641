import time
import pandas as pd
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



train_size_list = []
time_train_list = []
predict_train_list = []
accuracy_list = []
num_nodes_list = []

for i in range(0,19):
    t_size = 1 + 50 * i
    nn_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1, max_iter = t_size)

    print '----------------------------------------------------------------'
    print 'TESTING SIZE: ', t_size

    startt = time.time()
    nn_mlp.fit(trained, trainLabel)
    endt = time.time()

    print 'Training Size', len(mnist_train)
    print 'Test Size', len(mnist_test)
    print 'Time to train MLP', endt - startt

    start = time.time()
    nn_pred_result = nn_mlp.predict(mnist_test)
    end = time.time()

    print 'Time to Predict with MLP', end - start

    train_size_list.append(len(mnist_train))
    time_train_list.append(endt - startt)
    predict_train_list.append(end - start)

    correct = 0
    total = 0
    for i,j in zip(nn_pred_result,mnist_label):
        if i == j:
            correct += 1.0
        total += 1.0

    print 'Accuracy', correct / total

    accuracy_list.append(correct / total)

finalReport = pd.DataFrame(data = train_size_list, columns=['Training Size'])
finalReport['Time to Train MLP'] = time_train_list
finalReport['Time to Predict with MLP'] = predict_train_list
finalReport['Accuracy'] = accuracy_list

finalReport.to_csv('./nn_mlpReport_MNIST.csv',index=False)

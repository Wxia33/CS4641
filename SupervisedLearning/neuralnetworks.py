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

nn_mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)

nn_mlp.fit(trained, trainLabel)

mnist_label = mnist_test.label
mnist_test = mnist_test.drop('label',1)

nn_pred_result = nn_mlp.predict(mnist_test)

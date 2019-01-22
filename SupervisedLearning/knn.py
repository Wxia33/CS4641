import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

data = './data/MNIST/'

# data obtained from https://www.kaggle.com/oddrationale/mnist-in-csv#mnist_test.csv
testMnist = data + 'mnist_test.csv'
trainMnist = data + 'mnist_train.csv'

mnist_train = pd.read_csv(trainMnist)
mnist_test = pd.read_csv(testMnist)

knnClass = KNeighborsClassifier(n_neighbors = 10)

trainLabel = mnist_train.label
trained = mnist_train.drop('label',1)

knnClass.fit(trained, trainLabel)

mnist_label = mnist_test.label
mnist_test = mnist_test.drop('label',1)

knnPredictResult = knnClass.predict(mnist_test)

print knnPredictResult

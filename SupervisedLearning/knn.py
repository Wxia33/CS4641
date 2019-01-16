from sklearn.neighbors import KNeighborsClassifier

data = './data/MNIST/'

# data obtained from https://www.kaggle.com/oddrationale/mnist-in-csv#mnist_test.csv
test = data + 'mnist_test.csv'
train = data + 'mnist_train.csv'

knnClass = KNeighborsClassifier(n_neighbors=3)
knnClass.fit(X, y)

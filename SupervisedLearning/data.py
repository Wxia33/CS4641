import numpy as np
import pandas as pd
from sklearn import model_selection

data = './data/'

#Data obtained from https://www.kaggle.com/mlg-ulb/creditcardfraud
credCard = data + 'creditCardTest.csv'
credCard = pd.read_csv(credCard)


trainLabel = credCard.Class
trained = credCard.drop('Class',1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(trained, trainLabel, test_size = 85000, random_state = 42)

X_train['Class'] = y_train
X_test['Class'] = y_test

X_train.to_csv('./creditCardTrain.csv',index=False)
X_test.to_csv('./creditCardTest.csv',index=False)

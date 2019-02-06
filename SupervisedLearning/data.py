import numpy as np
import pandas as pd
from sklearn import model_selection

data = './data/'

preprocessData = 1

# Split Credit Card Data into training and test
if preprocessData == 0:
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

if preprocessData == 1:
    wineRed = data + 'winequality-red.csv'
    wineWhite = data + 'winequality-white.csv'

    wineRed = pd.read_csv(wineRed)
    wineWhite = pd.read_csv(wineWhite)

    #trainLabelRed = wineRed.quality

    title = 'fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'
    newTitle = title.replace('"','')
    newTitle = newTitle.split(';')

    splitWineRed = wineRed[title].str.split(';',expand=True)
    splitWineWhite = wineWhite[title].str.split(';',expand=True)

    totWineDat = pd.concat([splitWineRed,splitWineWhite])

    wineQualityLabel = totWineDat[11]
    wineQualTrain = totWineDat.drop(11,1)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(wineQualTrain, wineQualityLabel, test_size = 1000, random_state = 42)

    X_train[11] = y_train
    X_test[11] = y_test

    X_train.to_csv('./wineQualityTrain.csv', index = False)
    X_test.to_csv('./wineQualityTest.csv', index = False)

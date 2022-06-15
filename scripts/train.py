import numpy as np 
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import feature_engineering

def regression (dataframe):
    X = dataframe.drop('SalePrice',axis=1).copy()
    y = dataframe['SalePrice']
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    reg = LinearRegression().fit(xtrain, ytrain)
    ypred = reg.predict(xtest)
    offline_value = reg.score(X, y)
    return print("Offline value "+str(offline_value))

if __name__ == "__main__":
    train_size = 0.8
    test_size=0.2
    regression(feature_engineering.df_c_train_encoding)


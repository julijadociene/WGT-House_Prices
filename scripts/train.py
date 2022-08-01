import numpy as np 
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from feature_engineering import removing_missing_values
from feature_engineering import encoding
from feature_engineering import df_c_train
from feature_engineering import df_c_test

df_c_train_encoding = encoding(df_c_train)
df_c_test_encoding = encoding(df_c_test)

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
    regression(df_c_train_encoding)


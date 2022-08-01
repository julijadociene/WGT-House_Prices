import pandas as pd
import numpy as np
import subprocess
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
#from typing import list -> veikia tik su senesn4mis python versijomis...

#Defining dataframe with columns with missing data
def missing_cols(dataframe):
    missing_df = {} 

    for column in dataframe.columns:
        if dataframe[column].isna().sum() > 0:
            missing_df[column] = dataframe[column]

    # converting the missing_df to a dataframe
    missing_df = pd.DataFrame(missing_df, index = ['MissingValues']).T.sort_values(by='MissingValues', ascending=False)
    return missing_df

#Simple Imputer for missing values
def removing_missing_values(dataframe:pd.DataFrame, column_names_list, strategy, fill_value) -> pd.DataFrame:
    for column in column_names_list:
        imputer = SimpleImputer(missing_values = np.nan, strategy=strategy, fill_value=fill_value)
        imputer.fit(dataframe.loc[:, column_names_list])
        dataframe.loc[:, column_names_list] = imputer.transform(dataframe.loc[:, column_names_list])
    return dataframe

#Making dummy variables
def encoding(dataframe):
    #make dataframe copy
    dataframe_encoding = dataframe.copy()
    dataframe_encoding = pd.get_dummies(dataframe)
    return dataframe_encoding

def train_test_sample(dataframe):
    X = dataframe.drop('SalePrice',axis=1).copy()
    y = dataframe['SalePrice']
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    return xtrain, xtest, ytrain, ytest, X, y

def regression(xtrain, xtest, ytrain, X, y):
    reg = LinearRegression().fit(xtrain, ytrain)
    ypred = reg.predict(xtest)
    offline_value = reg.score(X, y)
    return print("Regression R^2: "+str(offline_value))

def GBregessor(random_state, loss, learning_rate, xtrain, xtest, ytrain):
    regXGB = GradientBoostingRegressor(random_state=random_state, loss=loss, learning_rate=learning_rate)
    regXGB.fit(xtrain, ytrain)
    y_predXGB = regXGB.predict(xtest)
    f=regXGB.score(xtrain, ytrain)
    return print("Gradient Boosting R^2: ", f)

def Random_Forest(max_depth, xtrain, ytrain):
    modelRFC = RandomForestClassifier(max_depth=max_depth)
    modelRFC.fit(xtrain,ytrain)
    preds = modelRFC.predict(xtrain)
    return print('Random Forest R^2: ', sklearn.metrics.r2_score(ytrain,preds))

#pathai, kintamieji, funkcijų vykdymas
if __name__ == "__main__":
    df_train = pd.read_csv(r'.\data\train.csv')
    missing_df_train = missing_cols(df_train)
    # creating columns depending on the imputing value for missing values
    col_na_to_None = ['Alley',
                        'BsmtQual',
                        'BsmtCond',
                        'BsmtExposure',
                        'BsmtFinType1',
                        'BsmtFinType2',
                        'FireplaceQu',
                        'GarageType',
                        'GarageFinish',
                        'GarageQual',
                        'GarageCond',
                        'PoolQC',
                        'MiscFeature']
    col_na_to_mf = ['MasVnrType',
                        'Electrical',
                        'Fence']
    col_na_to_avg = ['LotFrontage',
                        'MasVnrArea',
                        'GarageYrBlt']
    df_c_train = removing_missing_values(df_train, col_na_to_None, 'constant', 'None')
    df_c_train = removing_missing_values(df_train, col_na_to_mf, 'most_frequent', 'None')
    df_c_train = removing_missing_values(df_train, col_na_to_avg, 'mean', 'None')              

    df_test = pd.read_csv(r'.\data\test.csv')
    missing_df_test = missing_cols(df_test)
    columns_na_to_None_test = ['Alley',
                      'BsmtQual',
                      'BsmtCond',
                      'BsmtExposure',
                      'BsmtFinType1',
                      'BsmtFinType2',
                      'FireplaceQu',
                      'GarageType',
                      'GarageFinish',
                      'GarageQual',
                      'GarageCond',
                      'PoolQC',
                      'MiscFeature']
    columns_na_to_mf_test = ['MSZoning',
                    'Utilities',
                    'Exterior1st',
                    'Exterior2nd',
                    'MasVnrType',
                    'Electrical',
                    'KitchenQual',
                    'Functional',
                    'Fence',
                    'SaleType']
    columns_na_to_avg_test = ['LotFrontage',
                     'MasVnrArea',
                     'BsmtFinSF1',
                     'BsmtFinSF2',
                     'BsmtUnfSF',
                     'TotalBsmtSF',
                     'GarageYrBlt']
    columns_na_to_0_test = ['BsmtFullBath',
                   'BsmtHalfBath',
                   'GarageCars',
                   'GarageArea']
df_c_test = removing_missing_values(df_test, columns_na_to_None_test, 'constant', 'None')
df_c_test = removing_missing_values(df_test, columns_na_to_mf_test, 'most_frequent', 'None')
df_c_test = removing_missing_values(df_test, columns_na_to_avg_test, 'mean', 'None')
df_c_test = removing_missing_values(df_test, columns_na_to_0_test, 'constant', 0)

df_c_train_encoding = encoding(df_c_train)
df_c_test_encoding = encoding(df_c_test)

xtrain, xtest, ytrain, ytest, X, y = train_test_sample(df_c_train_encoding)
regression(xtrain, xtest, ytrain, X, y)
GBregessor(42, 'ls', 0.1, xtrain, xtest, ytrain)
Random_Forest(20, xtrain, ytrain)

#best model Random Forest, thus using it with test data
df_c_test_encoding_low = df_c_test_encoding.sample(n=1168) #getting the right shape for df
Random_Forest(20, df_c_test_encoding_low, ytrain)


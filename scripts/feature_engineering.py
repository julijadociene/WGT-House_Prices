import pandas as pd
import numpy as np
import subprocess
import sklearn
from sklearn.impute import SimpleImputer

#Defining dataframe with columns with missing data
def missing_cols(dataframe):
    missing_df = {} 

    for column in dataframe.columns:
        if dataframe[column].isna().sum() > 0:
            missing_df[column] = dataframe[column]

    # converting the missing_df to a dataframe
    missing_df = pd.DataFrame(missing_df, index = ['MissingValues']).T.sort_values(by='MissingValues', ascending=False)
    return missing_df

#Simple Imputer for missing values train
def removing_missing_values(dataframe):
   for column in col_na_to_None:
      imputer = SimpleImputer(missing_values = np.nan, strategy='constant', fill_value='None')
      imputer.fit(dataframe.loc[:, col_na_to_None])
      dataframe.loc[:, col_na_to_None] = imputer.transform(dataframe.loc[:, col_na_to_None])
   for column in col_na_to_mf:
      imputer2 = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
      imputer2.fit(dataframe.loc[:, col_na_to_mf])
      dataframe.loc[:, col_na_to_mf] = imputer2.transform(dataframe.loc[:, col_na_to_mf])
   for column in col_na_to_avg:
      imputer3 = SimpleImputer(missing_values = np.nan, strategy='mean')
      imputer3.fit(dataframe.loc[:, col_na_to_avg])
      dataframe.loc[:, col_na_to_avg] = imputer3.transform(dataframe.loc[:, col_na_to_avg])
   return dataframe

#Simple Imputer for missing values test
def removing_missing_values_test(dataframe):
   for column in columns_na_to_None_test:
      imputer4 = SimpleImputer(missing_values = np.nan, strategy='constant', fill_value='None')
      imputer4.fit(dataframe.loc[:, columns_na_to_None_test])
      dataframe.loc[:, columns_na_to_None_test] = imputer4.transform(dataframe.loc[:, columns_na_to_None_test])
   for column in columns_na_to_mf_test:
      imputer5 = SimpleImputer(missing_values = np.nan, strategy='most_frequent')
      imputer5.fit(dataframe.loc[:, columns_na_to_mf_test])
      dataframe.loc[:, columns_na_to_mf_test] = imputer5.transform(dataframe.loc[:, columns_na_to_mf_test])
   for column in columns_na_to_avg_test:
      imputer6 = SimpleImputer(missing_values = np.nan, strategy='mean')
      imputer6.fit(dataframe.loc[:, columns_na_to_avg_test])
      dataframe.loc[:, columns_na_to_avg_test] = imputer6.transform(dataframe.loc[:, columns_na_to_avg_test])
   for column in columns_na_to_0_test:
      imputer7 = SimpleImputer(missing_values = np.nan, strategy='constant', fill_value = 0)
      imputer7.fit(dataframe.loc[:, columns_na_to_0_test])
      dataframe.loc[:, columns_na_to_0_test] = imputer7.transform(dataframe.loc[:, columns_na_to_0_test])
   return dataframe

#Making dummy variables
def encoding(dataframe):
    #make dataframe copy
    dataframe_encoding = dataframe.copy()
    dataframe_encoding = pd.get_dummies(dataframe)
    return dataframe_encoding

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
    df_c_train = removing_missing_values(df_train)
    
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
    df_c_test = removing_missing_values_test(df_test)

df_c_train_encoding = encoding(df_c_train)
df_c_test_encoding = encoding(df_c_test)
    

#galimi taisymai:
#1. removing_missing_values viena funkcija train ir test df
#2. function to generate missing columns to list (e.g. columns_na_to_None)



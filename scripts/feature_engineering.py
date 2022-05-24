import pandas as pd
import numpy as np
import subprocess
import os
import os.path
import matplotlib.pyplot as plt

#import train.csv file
df = pd.read_csv(r'.\data\train.csv')

#Feature Engineering
#Checking for missing values
#print(df.isna())
#print(df.isna().any())
#print(df.isna().sum().plot(kind="bar"))
#plt.show

def missing_cols():
    missing_df = {} #dictionary???

    for column in df.columns:
        if df[column].isna().sum() > 0:
            missing_df[column] = df[column]

### kodel reikia versti į dataframe? ar i lista pradzioj viska sudeda???
    # converting the missing_df to a dataframe
    missing_df = pd.DataFrame(missing_df, index = ['MissingValues']).T.sort_values(by='MissingValues', ascending=False)
    return missing_df

missing_df = missing_cols()
print(missing_df)

#fill missing values with mean/0

##ERROR "TypeError: string indices must be integers", in fill_na_values(), how to handle it?
##I tried to generate mean values seperately for each column and then fill it, doesnt work :/

#create empty dataframe for means
df_mean = pd.DataFrame()

#generate column mean values in df_mean
def df_mean_values():
    for column in df:
        if df[column].dtype == 'float64' or 'int64': 
            df_mean[column] = df[column].mean
        else:
            #gal kazkaip kitaip? arba dažniausiai pasitaikančia reikšmę? Kaip?
            df_mean[column] = 0
    return df_mean

df_mean = df_mean_values()
print(df_mean)

#fill na values in df
def fill_na_values():

    for column in df.columns:
        if df[column].isna().sum() > 0:
            for value in column:
                df[column[value]] = df_mean[column]
    return df

df_new = fill_na_values()
print(df_new)


#Tekstiniai kintamieji į str



#Atskirame faile?
#which columns should I include in regression?
 #To find the parameters, we need to minimize the least squares or the sum of squared errors
 # find coefficients(parameters/betas)
 #to tell if coefficients are relevant to predict dependent variable find p-value (F-statistic?)
 #find RSE(the lower the residual errors, the better the model fits the data )/ R^2( proportion of variability in the target that can be explained using a feature X)

#train data 




import pandas as pd
import numpy as np
import subprocess
from scripts import unzip_data
import os
import os.path



#save output to results location
    #first arg is the name of the file
    #second arg notes that the file is open to write to it
with open(r'.\results\readme.txt', 'w') as f:
    
    #unzip downloaded data
    subprocess.call("unzip_data.py", shell=True)

    #delete house-prices-advanced-regression-techniques.zip file
    os.remove("house-prices-advanced-regression-techniques.zip")

    #import train.csv file
    df = pd.read_csv(r'.\data\train.csv')
    #df.head(df)

    #Feature Engineering
    #Checking for missing values
    print(df.isnull().count)


#Rezultatų masyvas 
    df_results = 1

#write results to '.\results\results.txt'
    f.write(df_results)

import zipfile

with zipfile.ZipFile('.\house-prices-advanced-regression-techniques.zip', 'r') as zip_ref:
    zip_ref.extractall('.\data')


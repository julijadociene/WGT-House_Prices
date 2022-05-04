dvc stage add -n "download data" kaggle competitions download -c house-prices-advanced-regression-techniques

dvc stage add -n "unzip data" with zipfile.ZipFile('.\house-prices-advanced-regression-techniques.zip', 'r') as zip_ref:
    zip_ref.extractall('.\data')

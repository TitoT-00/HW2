import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np

#Preprocessing ...
def preprocess(data, is_train=True):
    data = data.drop(columns=['Alley','PoolQc','Fence','MiscFeature'], errors='ignore')

      # Fill missing values with median for numerical features and mode for categorical features
    for col in data.select_dtypes(include=['float64', 'int64']).columns:
        data[col].fillna(data[col].median(), inplace=True)
    for col in data.select_dtypes(include=['object']).columns:
        data[col].fillna(data[col].mode()[0], inplace=True)

    # One-hot encode categorical variables
    categorical_cols = data.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))
    encoded_cols.columns = encoder.get_feature_names_out(categorical_cols)
    data = pd.concat([data.drop(columns=categorical_cols), encoded_cols], axis=1)

    # Remove SalePrice for test data
    if not is_train:
        data = data.drop(columns=['SalePrice'], errors='ignore')

    return data



def main():
    # Loading data ... 
    train_data = pd.read_csv('/train.csv')
    test_data = pd.read_csv('/test.csv')

    # Preprocessing ... 
    train_data_preprocessing = preprocess(train_data)
    test_data_preprocessing = preprocess(test_data, is_train=False)

    # Split features and target for training
    y = train_data_preprocessing['SalePrice']
    x = train_data_preprocessing.drop(columns=['SalePrice'])

    #Train-test split for validation
    X_train, X_val, y_train, y_val = 


if __name__ == "main":
  main()  
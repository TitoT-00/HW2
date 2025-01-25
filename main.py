import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_features(data):
    """Create new features based on domain knowledge"""
    data = data.copy()
    
    # Total square footage
    data['TotalSF'] = data['1stFlrSF'] + data['2ndFlrSF'] + data['TotalBsmtSF']
    
    # Total bathrooms
    data['TotalBathrooms'] = data['FullBath'] + 0.5*data['HalfBath'] + data['BsmtFullBath'] + 0.5*data['BsmtHalfBath']
    
    # House age and renovation
    data['Age'] = 2025 - data['YearBuilt']
    data['YearsSinceRemodel'] = 2025 - data['YearRemodAdd']
    
    return data

def preprocess(data, preprocessor=None, is_train=True):
    """Preprocess the data with proper handling of different feature types"""
    data = data.copy()
    
    # Drop columns with too many missing values
    data = data.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature'], errors='ignore')
    
    # Create new features
    data = create_features(data)
    
    # Define feature groups
    numerical_features = [
        'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 
        'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
        'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch',
        'TotalSF', 'TotalBathrooms', 'Age', 'YearsSinceRemodel',
        'OverallQual', 'OverallCond'  # Adding these as numerical features
    ]
    
    categorical_features = [
        'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
        'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
        'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'Foundation', 'Heating', 'CentralAir', 'Functional',
        'GarageType', 'SaleType', 'SaleCondition'
    ]
    
    # Fill missing values
    for col in numerical_features:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    for col in categorical_features:
        if col in data.columns:
            data[col] = data[col].fillna('Missing')
    
    # Convert quality features to numeric
    quality_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                       'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']
    
    quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'Missing': 0, np.nan: 0}
    for col in quality_features:
        if col in data.columns:
            data[col] = data[col].map(quality_map)
            numerical_features.append(col)
    
    if preprocessor is None and is_train:
        # Initialize transformers
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Fit and transform
        X = preprocessor.fit_transform(data)
        
        if not is_train:
            return X, preprocessor
        return X, preprocessor
    
    else:
        # Transform using existing preprocessor
        X = preprocessor.transform(data)
        return X

def main():
    # Load data
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    # Preprocess data
    X_train, preprocessor = preprocess(train_data, is_train=True)
    y_train = train_data['SalePrice']
    X_test = preprocess(test_data, preprocessor=preprocessor, is_train=False)
    
    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_split, y_train_split)
    
    # Validate model
    val_predictions = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"Validation RMSE: {rmse}")
    
    # Make predictions on test data
    test_predictions = model.predict(X_test)
    
    # Save predictions
    submission = pd.DataFrame({
        'Id': test_data['Id'],
        'SalePrice': test_predictions
    })
    submission.to_csv('test_predictions.csv', index=False)
    print("Predictions saved to test_predictions.csv")

if __name__ == "__main__":
    main()
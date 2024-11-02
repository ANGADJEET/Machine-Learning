import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(file_path):
    # Load the dataset
    return pd.read_csv(file_path)

def handle_missing_values(df):
    # check for missing values
    missing_values_count = df.isnull().sum()
    print(f"Total missing values in the dataset: {missing_values_count.sum()}")
    return df

def standardize_features(df):
    # Standardize the features
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return scaled_df

def encode_categorical_features(df):
    # Encode categorical features
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}
    df_encoded = df.copy()
    for feature in categorical_features:
        le = LabelEncoder()
        df_encoded[feature] = le.fit_transform(df_encoded[feature])
        label_encoders[feature] = le
    return df_encoded, label_encoders

def split_data(df, target_column, test_size=0.2, random_state=42):
    # Split the data into features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

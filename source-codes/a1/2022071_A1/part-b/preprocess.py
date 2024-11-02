import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_and_preprocess(file_path):
    dataset = pd.read_csv(file_path)
    # Handle missing values by filling with the median
    dataset.fillna(dataset.mean(), inplace=True)
    return dataset

def prepare_data(dataset, target_column):
    # Split the dataset into features and target
    features = dataset.drop(target_column, axis=1)
    target = dataset[target_column]
    return features, target

def split_data(features, target, test_ratio=0.3, val_ratio=0.5, random_seed=42):
    # Split the data into train, validation and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=test_ratio, random_state=random_seed, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_ratio, random_state=random_seed, shuffle=True)
    return X_train, X_val, X_test, y_train, y_val, y_test

def apply_smote(X_train, y_train, random_seed=42):
    # Apply SMOTE to balance the classes
    smote = SMOTE(random_state=random_seed)
    return smote.fit_resample(X_train, y_train)

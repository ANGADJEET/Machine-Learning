import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from tabulate import tabulate
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import FastICA
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
import os

def create_directory(directory):
    """Create a directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def plot_pairwise_relationships(df, directory='plots', filename_prefix='pairwise_relationships'):
    """Generate a pair plot for numerical features and save it as a PNG file with 300 dpi in the specified directory."""
    # Ensure the directory exists
    create_directory(directory)
    
    # Generate the pair plot
    pair_plot = sns.pairplot(df.select_dtypes(include=[np.number, np.float64]))
    plt.suptitle("Pair Plot of Numerical Features", y=1.02)
    
    # Save the plot with 300 dpi
    file_path = os.path.join(directory, f'{filename_prefix}.png')
    
    # Save the plot with 300 dpi
    pair_plot.savefig(file_path, dpi=300)
    
    # Show the plot
    plt.show()
    
    
def do_one_hot_encoding(df, categorical_columns):
    # Initialize the OneHotEncoder with updated parameters
    one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')

    # Fit and transform the categorical columns
    encoded_features = one_hot_encoder.fit_transform(df[categorical_columns])

    # Create proper column names for the encoded features
    encoded_feature_names = one_hot_encoder.get_feature_names_out(categorical_columns)

    # Create a DataFrame with the encoded features and correct column names
    df_encoded = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=df.index)

    # Drop the original categorical columns from the DataFrame
    df.drop(columns=categorical_columns, inplace=True)

    # Concatenate the original DataFrame with the encoded DataFrame
    df_final = pd.concat([df, df_encoded], axis=1)

    return df_final

def plot_box_plots(df, numeric_columns, directory='plots', filename_prefix='box_plot'):
    """Create and save box plots for numerical columns."""
    n_cols = 3
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten()

    for i, column in enumerate(numeric_columns):
        sns.boxplot(x=df[column], ax=axes[i])
        axes[i].set_title(f"Box Plot of {column}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    # Ensure the directory exists
    create_directory(directory)
    
    # Save the plot with 300 dpi
    file_path = os.path.join(directory, f'{filename_prefix}.png')
    plt.savefig(file_path, dpi=300)
    
    plt.show()

def plot_violin_plots(df, numeric_columns, directory='plots', filename_prefix='violin_plot'):
    """Create and save violin plots for numerical columns."""
    n_cols = 3
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.flatten()

    for i, column in enumerate(numeric_columns):
        sns.violinplot(x=df[column], ax=axes[i])
        axes[i].set_title(f"Violin Plot of {column}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    # Ensure the directory exists
    create_directory(directory)
    
    # Save the plot with 300 dpi
    file_path = os.path.join(directory, f'{filename_prefix}.png')
    plt.savefig(file_path, dpi=300)
    
    plt.show()

def plot_count_plots(df, categorical_columns, directory='plots', filename_prefix='count_plot'):
    """Create and save count plots for categorical features."""
    n_cols = 3
    n_rows = (len(categorical_columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 5))
    axes = axes.flatten()

    for i, column in enumerate(categorical_columns):
        sns.countplot(x=df[column], ax=axes[i])
        axes[i].set_title(f"Count Plot of {column}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    
    # Ensure the directory exists
    create_directory(directory)
    
    # Save the plot with 300 dpi
    file_path = os.path.join(directory, f'{filename_prefix}.png')
    plt.savefig(file_path, dpi=300)
    
    plt.show()

def plot_correlation_matrices(original_df, encoded_df, target_column, directory='plots', filename_prefix='correlation_matrix'):
    """Plot and save correlation matrices for original and encoded data."""
    non_categorical_features = original_df.drop(columns=[target_column])
    original_corr = non_categorical_features.corr()
    encoded_corr = encoded_df.corr()

    fig, axes = plt.subplots(1, 2, figsize=(24, 10))
    sns.heatmap(original_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[0])
    axes[0].set_title("Correlation Matrix (Original Data)")

    sns.heatmap(encoded_corr, annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1])
    axes[1].set_title("Correlation Matrix (Encoded Data)")

    plt.tight_layout()
    
    # Ensure the directory exists
    create_directory(directory)
    
    # Save the plot with 300 dpi
    file_path = os.path.join(directory, f'{filename_prefix}.png')
    plt.savefig(file_path, dpi=300)
    
    plt.show()

# Plot violin plots for numerical features
# Function for UMAP dimensionality reduction
def do_umap(df, dataset):
    reducer = umap.UMAP(n_components=2, random_state=42)
    embedding = reducer.fit_transform(df)
    
    df_umap = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    df_umap['Electricity_Bill'] = dataset['Electricity_Bill'].values
    return df_umap
  
def plot_umap(df_unscaled_unencoded, target, directory='plots', filename_prefix='umap_plot'):
    """Plot UMAP results for numerical and encoded data and save it as a PNG file with 300 dpi."""
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=df_unscaled_unencoded.iloc[:, 0], 
        y=df_unscaled_unencoded.iloc[:, 1], 
        hue=target, 
        palette=sns.color_palette("hsv", as_cmap=True)
    )
    
    plt.title("UMAP Visualization of Data")
    
    create_directory(directory)
    # Save the plot with 300 dpi
    file_path = os.path.join(directory, f'{filename_prefix}.png')
    plt.savefig(file_path, dpi=300)
    
    plt.show()
    
def train_linear_regression(X_train, y_train):
    """Train a Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
  
def calculate_adjusted_r2(r2, n, p):
    """Calculate Adjusted R²."""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def perform_feature_selection(X_train, y_train, num_features=3):
    """Select important features using RFE or SelectKBest."""
    rfe_selector = RFE(LinearRegression(), n_features_to_select=num_features)
    rfe_selector.fit(X_train, y_train)
    rfe_features = X_train.columns[rfe_selector.support_]
    
    kbest_selector = SelectKBest(score_func=f_regression, k=num_features)
    kbest_selector.fit(X_train, y_train)
    kbest_features = X_train.columns[kbest_selector.get_support()]

    return rfe_features, kbest_features

def show_metrics(model, X_train, y_train, X_test, y_test):
    # Predicting on train and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate evaluation metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    train_adjusted_r2 = calculate_adjusted_r2(train_r2, X_train.shape[0], X_train.shape[1])
    test_adjusted_r2 = calculate_adjusted_r2(test_r2, X_test.shape[0], X_test.shape[1])

    # Prepare data for the table
    metrics = [
        ["Metric", "Training", "Testing"],
        ["MAE", train_mae, test_mae],
        ["MSE", train_mse, test_mse],
        ["RMSE", train_rmse, test_rmse],
        ["R2", train_r2, test_r2],
        ["Adjusted R2", train_adjusted_r2, test_adjusted_r2]
    ]

    # Print the table using tabulate
    print(tabulate(metrics[1:], headers=metrics[0], tablefmt="grid", floatfmt=".5f"))
 
# Function to perform ICA
def do_ica(X, n_components):
    ica = FastICA(n_components=n_components, random_state=42)
    X_ica = ica.fit_transform(X)
    X_ica = pd.DataFrame(X_ica, columns=[f'ICA{i+1}' for i in range(n_components)])
    return X_ica

# Function to train the Ridge Regression model
def train_ridge(X_train, y_train, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model
   
# Main function to perform ICA and Ridge regression for different components
def train_ridge_along_with_ica(X_train, X_test, y_train, y_test, n_components_list):
    for n_components in n_components_list:
        print(f"\nPerforming ICA with {n_components} components")
        
        # Perform ICA on train and test sets
        X_train_ica = do_ica(X_train, n_components)
        X_test_ica = do_ica(X_test, n_components)
        
        # Train Ridge Regression model
        model = train_ridge(X_train_ica, y_train)
        
        y_train_pred = model.predict(X_train_ica)
        y_test_pred = model.predict(X_test_ica)
    
        print(f"\nData Metrics with ICA (n_components={n_components})")
        show_metrics(model=model, X_train=X_train_ica, y_train=y_train, X_test=X_test_ica, y_test=y_test)

def do_elastic_net(X_train, y_train, alpha=1.0, l1_ratio=0.5):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model
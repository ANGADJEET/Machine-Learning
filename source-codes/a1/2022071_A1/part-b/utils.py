import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import seaborn as sns
import os

# Min-Max Scaling
def min_max_function_formula(X, min_values, max_values):
    return (X - min_values) / (max_values - min_values)

def apply_min_max_scaling(train_features, val_features, test_features):
    feature_max = train_features.max(axis=0)
    feature_min = train_features.min(axis=0)
    
    train_scaled = min_max_function_formula(train_features, feature_min, feature_max)
    val_scaled = min_max_function_formula(val_features, feature_min, feature_max)
    test_scaled = min_max_function_formula(test_features, feature_min, feature_max)
    return train_scaled, val_scaled, test_scaled



def create_plots_directory():
    if not os.path.exists('plots'):
        os.makedirs('plots')


# Plotting functions
def make_plots(train_losses, val_losses, train_accuracies, val_accuracies, method_name):
    create_plots_directory()
    
    plt.figure(figsize=(12, 6))
    
    # Loss vs Epoch
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label=f'Train Loss ({method_name})', color='blue')
    plt.plot(val_losses, label=f'Val Loss ({method_name})', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss over Epochs ({method_name})')
    plt.legend()

    # Accuracy vs Epoch
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label=f'Train Accuracy ({method_name})', color='blue')
    plt.plot(val_accuracies, label=f'Val Accuracy ({method_name})', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy over Epochs ({method_name})')
    plt.legend()

    plt.tight_layout()
    

    # Save the plot in the 'plots' directory
    plt.savefig(f'plots/{method_name}_plot.png', dpi=300)
    
    plt.show()

def make_comparison_plots(train_losses1, val_losses1, train_accuracies1, val_accuracies1, method_name1,
                          train_losses2, val_losses2, train_accuracies2, val_accuracies2, method_name2):
    create_plots_directory()
    
    plt.figure(figsize=(12, 6))
    
    # Loss vs Epoch
    plt.subplot(1, 2, 1)
    plt.plot(train_losses1, label=f'Train Loss ({method_name1})', color='blue')
    plt.plot(val_losses1, label=f'Val Loss ({method_name1})', color='orange')
    plt.plot(train_losses2, label=f'Train Loss ({method_name2})', color='green')
    plt.plot(val_losses2, label=f'Val Loss ({method_name2})', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss over Epochs ({method_name1} vs {method_name2})')
    plt.legend()

    # Accuracy vs Epoch
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies1, label=f'Train Accuracy ({method_name1})', color='blue')
    plt.plot(val_accuracies1, label=f'Val Accuracy ({method_name1})', color='orange')
    plt.plot(train_accuracies2, label=f'Train Accuracy ({method_name2})', color='green')
    plt.plot(val_accuracies2, label=f'Val Accuracy ({method_name2})', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy over Epochs ({method_name1} vs {method_name2})')
    plt.legend()

    plt.tight_layout()
    # Save the plot in the 'plots' directory
    plt.savefig(f'plots/{method_name1}_vs_{method_name2}_plot.png', dpi=300)
    
    plt.show()

def compare_4_plots(train_losses1, val_losses1, train_accuracies1, val_accuracies1, method_name1, train_losses2, val_losses2, train_accuracies2, val_accuracies2, method_name2,
                    train_losses3, val_losses3, train_accuracies3, val_accuracies3, method_name3, train_losses4, val_losses4, train_accuracies4, val_accuracies4, method_name4):
    
    create_plots_directory()
    plt.figure(figsize=(12, 12))
    
    # Put training loss plots
    plt.subplot(2, 2, 1)
    plt.plot(train_losses1, label=f'Train Loss ({method_name1})', color='blue')
    plt.plot(train_losses2, label=f'Train Loss ({method_name2})', color='green')
    plt.plot(train_losses3, label=f'Train Loss ({method_name3})', color='red')
    plt.plot(train_losses4, label=f'Train Loss ({method_name4})', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Train Loss over Epochs ({method_name1} vs {method_name2} vs {method_name3} vs {method_name4})')
    
    # Put validation loss plots
    plt.subplot(2, 2, 2)
    plt.plot(val_losses1, label=f'Val Loss ({method_name1})', color='blue')
    plt.plot(val_losses2, label=f'Val Loss ({method_name2})', color='green')
    plt.plot(val_losses3, label=f'Val Loss ({method_name3})', color='red')
    plt.plot(val_losses4, label=f'Val Loss ({method_name4})', color='purple')
    plt.xlabel('Epoch')
    
    plt.ylabel('Loss')
    plt.title(f'Val Loss over Epochs ({method_name1} vs {method_name2} vs {method_name3} vs {method_name4})')
    
    # Put training accuracy plots
    plt.subplot(2, 2, 3)
    plt.plot(train_accuracies1, label=f'Train Accuracy ({method_name1})', color='blue')
    plt.plot(train_accuracies2, label=f'Train Accuracy ({method_name2})', color='green')
    plt.plot(train_accuracies3, label=f'Train Accuracy ({method_name3})', color='red')
    plt.plot(train_accuracies4, label=f'Train Accuracy ({method_name4})', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.title(f'Train Accuracy over Epochs ({method_name1} vs {method_name2} vs {method_name3} vs {method_name4})')
    
    
    # Put validation accuracy plots
    
    plt.subplot(2, 2, 4)
    plt.plot(val_accuracies1, label=f'Val Accuracy ({method_name1})', color='blue')
    plt.plot(val_accuracies2, label=f'Val Accuracy ({method_name2})', color='green')
    plt.plot(val_accuracies3, label=f'Val Accuracy ({method_name3})', color='red')
    plt.plot(val_accuracies4, label=f'Val Accuracy ({method_name4})', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Val Accuracy over Epochs ({method_name1} vs {method_name2} vs {method_name3} vs {method_name4})')
    
    plt.tight_layout()
    
    # Save the plot in the 'plots' directory
    plt.savefig(f'plots/{method_name1}_vs_{method_name2}_vs_{method_name3}_vs_{method_name4}_plot.png', dpi=300)
    plt.show()
    
def compute_metrics(y_true, y_pred):
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Precision, Recall, F1 Score, and ROC-AUC Score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    
    return conf_matrix, precision, recall, f1, roc_auc

# Print metrics
def print_metrics(conf_matrix, precision, recall, f1, roc_auc):
    print("Confusion Matrix:")
    print(conf_matrix)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")

# Plot confusion matrix
def plot_confusion_matrix(conf_matrix, filename='confusion_matrix.png'):
    plt.figure(figsize=(10, 7))
    labels = ['Negative', 'Positive']  # Change labels to 'Negative' and 'Positive'
    
    # Annotate confusion matrix
    conf_matrix_text = np.empty(conf_matrix.shape, dtype=object)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            conf_matrix_text[i, j] = f'{conf_matrix[i, j]}'
    
    sns.heatmap(conf_matrix, annot=conf_matrix_text, fmt='', cmap='Blues',
                xticklabels=labels, yticklabels=labels, cbar=False,
                annot_kws={"size": 12, "weight": "bold"}, linewidths=.5)
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(filename)
    plt.show()
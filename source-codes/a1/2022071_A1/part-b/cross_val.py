import numpy as np
from model import LogisticRegressionModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabulate import tabulate


def kfoldcrossval(X, y, k=5, learning_rate=0.00001, epochs=30000):
    N, D = X.shape
    fold_size = N // k
    indices = np.arange(N)
    #for reproducibility
    np.random.shuffle(indices)

    accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []
    for i in range(k):
        val_indices = indices[i * fold_size:(i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])

        X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
        y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]
        
        model = LogisticRegressionModel(learning_rate=learning_rate, epochs=epochs)
        model.train_batch_gradient_descent(X_train, y_train, X_val, y_val)

        y_val_pred = model.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_val_pred) 
        precision = precision_score(y_val, y_val_pred)
        recall = recall_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

    return accuracy_scores, precision_scores, recall_scores, f1_scores

def print_kfold_results(accuracy_scores, precision_scores, recall_scores, f1_scores, title):
    headers = ["Fold", "Accuracy", "Precision", "Recall", "F1 Score"]
    table = []
    for i in range(len(accuracy_scores)):
        table.append([i+1, accuracy_scores[i], precision_scores[i], recall_scores[i], f1_scores[i]])
    print(title)
    print(tabulate(table, headers=headers, tablefmt="grid"))

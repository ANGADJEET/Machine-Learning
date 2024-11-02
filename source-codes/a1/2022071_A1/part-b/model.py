import numpy as np
from sklearn.metrics import accuracy_score

class LogisticRegressionModel:
    def __init__(self, learning_rate=0.00001, epochs=100000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.parameters = None

    def add_intercept(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy(self, true_labels, predicted_probs):
        return -np.mean(true_labels * np.log(predicted_probs) + (1 - true_labels) * np.log(1 - predicted_probs))

    def gradient_step(self, X, true_labels, predicted_probs):
        N = X.shape[0]
        return np.dot(X.T, (predicted_probs - true_labels)) / N

    def train_batch_gradient_descent(self, features, target, val_features, val_target, plot_interval=10000):
        N, num_features = features.shape
        features = self.add_intercept(features)
        val_features = self.add_intercept(val_features)

        self.parameters = np.zeros(num_features + 1)  # Initialize weights with bias

        training_loss_history, validation_loss_history = [], []
        training_accuracy_history, validation_accuracy_history = [], []

        for epoch in range(self.epochs):
            # Training predictions
            train_pred_probs = self.sigmoid(np.dot(features, self.parameters))
            train_loss = self.binary_cross_entropy(target, train_pred_probs)
            train_accuracy = accuracy_score(target, (train_pred_probs >= 0.5).astype(int))

            # Validation predictions
            val_pred_probs = self.sigmoid(np.dot(val_features, self.parameters))
            val_loss = self.binary_cross_entropy(val_target, val_pred_probs)
            val_accuracy = accuracy_score(val_target, (val_pred_probs >= 0.5).astype(int))

            # Gradient descent step
            grad = self.gradient_step(features, target, train_pred_probs)
            self.parameters -= self.learning_rate * grad

            # Record history
            training_loss_history.append(train_loss)
            validation_loss_history.append(val_loss)
            training_accuracy_history.append(train_accuracy)
            validation_accuracy_history.append(val_accuracy)

            if epoch % plot_interval == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss}, Train Accuracy = {train_accuracy}, "
                      f"Val Loss = {val_loss}, Val Accuracy = {val_accuracy}")

        return training_loss_history, validation_accuracy_history, validation_loss_history, training_accuracy_history
    
    
    def l2_binary_cross_entropy(self, true_labels, predicted_probs, parameters, alpha):
        """L2 Regularized Binary Cross Entropy Loss"""
        return -np.mean(true_labels * np.log(predicted_probs) + (1 - true_labels) * np.log(1 - predicted_probs)) + alpha * np.sum(parameters[1:] ** 2)
    
    def l1_binary_cross_entropy(self, true_labels, predicted_probs, parameters, alpha):
        """L1 Regularized Binary Cross Entropy Loss"""
        return -np.mean(true_labels * np.log(predicted_probs) + (1 - true_labels) * np.log(1 - predicted_probs)) + alpha * np.sum(np.abs(parameters[1:]))
    
    def train_stochastic_gradient_descent_with_early_stopping(self, features, target, val_features, val_target,iter_without_improvement, alpha, regulariser=None, plot_interval=10000, tol=0.01):
        N, num_features = features.shape
        features = self.add_intercept(features)  # Add intercept term
        val_features = self.add_intercept(val_features)

        # Initialize parameters
        self.parameters = np.zeros(num_features + 1)  # +1 for the intercept
        
        # Lists to store history for plotting
        training_loss_history, validation_loss_history = [], []
        training_accuracy_history, validation_accuracy_history = [], []
        
        no_improvement = 0
        prev_val = 0
        
        for epoch in range(self.epochs):
            # Perform SGD
            #shuffle data
            idx = np.random.permutation(N)
            for i in range(N):
                # idx = np.random.randint(0, N)
                
                x_i = features[idx[i]]
                y_i = target[idx[i]]
                
                # Predicted probabilities using sigmoid
                train_pred_probs = self.sigmoid(np.dot(x_i, self.parameters))
                
                # Compute gradient
                grad = np.dot(x_i.T, (train_pred_probs - y_i))
                
                # Update parameters
                self.parameters -= self.learning_rate * grad
            
            # Compute training and validation metrics
            train_pred_probs = self.sigmoid(np.dot(features, self.parameters))
            train_accuracy = accuracy_score(target, (train_pred_probs >= 0.5).astype(int))
            
            val_pred_probs = self.sigmoid(np.dot(val_features, self.parameters))
            val_accuracy = accuracy_score(val_target, (val_pred_probs >= 0.5).astype(int))

            # Compute loss with regularization
            if regulariser == 'l2':
                train_loss = self.l2_binary_cross_entropy(target, train_pred_probs, self.parameters, alpha)
                val_loss = self.l2_binary_cross_entropy(val_target, val_pred_probs, self.parameters, alpha)
            elif regulariser == 'l1':
                train_loss = self.l1_binary_cross_entropy(target, train_pred_probs, self.parameters, alpha)
                val_loss = self.l1_binary_cross_entropy(val_target, val_pred_probs, self.parameters, alpha)
            else:
                train_loss = -np.mean(target * np.log(train_pred_probs) + (1 - target) * np.log(1 - train_pred_probs))
                val_loss = -np.mean(val_target * np.log(val_pred_probs) + (1 - val_target) * np.log(1 - val_pred_probs))
                        
            # # Logging
            # if epoch % plot_interval == 0:
            #     print(f"Epoch {epoch}: Train Loss = {train_loss}, Train Accuracy = {train_accuracy}, "
            #           f"Val Loss = {val_loss}, Val Accuracy = {val_accuracy}")
            # Early stopping
            # print("differece", abs(val_loss - prev_val))
            # print("current val", val_loss)
            # print("prev val", prev_val)
            if abs(val_loss - prev_val) > tol:
                prev_val = val_loss
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement == iter_without_improvement:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Append loss and accuracy to history
            training_loss_history.append(train_loss)
            validation_loss_history.append(val_loss)
            training_accuracy_history.append(train_accuracy)
            validation_accuracy_history.append(val_accuracy)
        
        return training_loss_history, validation_loss_history, training_accuracy_history, validation_accuracy_history
            
    def train_stochastic_gradient_descent(self, features, target, val_features, val_target, plot_interval=10000):
        N, num_features = features.shape
        features = self.add_intercept(features)
        val_features = self.add_intercept(val_features)

        self.parameters = np.zeros(num_features + 1)  # Initialize weights with bias

        training_loss_history, validation_loss_history = [], []
        training_accuracy_history, validation_accuracy_history = [], []

        for epoch in range(self.epochs):
            idx = np.random.permutation(N)
            for i in range(N):
                # idx = np.random.randint(0, N)
                
                x_i = features[idx[i]]
                y_i = target[idx[i]]

                # Training predictions
                train_pred_probs = self.sigmoid(np.dot(x_i, self.parameters))

                # Gradient descent step
                grad = np.dot(x_i.T, (train_pred_probs - y_i))
                self.parameters -= self.learning_rate * grad

            # Training predictions after epoch
            train_pred_probs = self.sigmoid(np.dot(features, self.parameters))
            train_loss = self.binary_cross_entropy(target, train_pred_probs)
            train_accuracy = accuracy_score(target, (train_pred_probs >= 0.5).astype(int))

            # Validation predictions after epoch
            val_pred_probs = self.sigmoid(np.dot(val_features, self.parameters))
            val_loss = self.binary_cross_entropy(val_target, val_pred_probs)
            val_accuracy = accuracy_score(val_target, (val_pred_probs >= 0.5).astype(int))

            # Record history
            training_loss_history.append(train_loss)
            validation_loss_history.append(val_loss)
            training_accuracy_history.append(train_accuracy)
            validation_accuracy_history.append(val_accuracy)

            if epoch % plot_interval == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss}, Train Accuracy = {train_accuracy}, "
                      f"Val Loss = {val_loss}, Val Accuracy = {val_accuracy}")

        return training_loss_history, validation_accuracy_history, validation_loss_history, training_accuracy_history

    def train_mini_batch_gradient_descent(self, features, target, val_features, val_target, batch_size=32, plot_interval=10000):
        N, num_features = features.shape
        features = self.add_intercept(features)
        val_features = self.add_intercept(val_features)

        self.parameters = np.zeros(num_features + 1)  # Initialize weights with bias

        training_loss_history, validation_loss_history = [], []
        training_accuracy_history, validation_accuracy_history = [], []

        for epoch in range(self.epochs):
            # Shuffle data
            indices = np.random.permutation(N)
            features_shuffled = features[indices]
            target_shuffled = target[indices]

            for i in range(0, N, batch_size):
                X_batch = features_shuffled[i:i + batch_size]
                y_batch = target_shuffled[i:i + batch_size]

                # Training predictions
                train_pred_probs = self.sigmoid(np.dot(X_batch, self.parameters))

                # Gradient descent step
                grad = np.dot(X_batch.T, (train_pred_probs - y_batch)) / batch_size
                self.parameters -= self.learning_rate * grad

            # Training predictions after epoch
            train_pred_probs = self.sigmoid(np.dot(features, self.parameters))
            train_loss = self.binary_cross_entropy(target, train_pred_probs)
            train_accuracy = accuracy_score(target, (train_pred_probs >= 0.5).astype(int))

            # Validation predictions after epoch
            val_pred_probs = self.sigmoid(np.dot(val_features, self.parameters))
            val_loss = self.binary_cross_entropy(val_target, val_pred_probs)
            val_accuracy = accuracy_score(val_target, (val_pred_probs >= 0.5).astype(int))

            # Record history
            training_loss_history.append(train_loss)
            validation_loss_history.append(val_loss)
            training_accuracy_history.append(train_accuracy)
            validation_accuracy_history.append(val_accuracy)

            if epoch % plot_interval == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss}, Train Accuracy = {train_accuracy}, "
                      f"Val Loss = {val_loss}, Val Accuracy = {val_accuracy}")

        return training_loss_history, validation_accuracy_history, validation_loss_history, training_accuracy_history

    def predict(self, features, threshold=0.5):
        features = self.add_intercept(features)
        predicted_probs = self.sigmoid(np.dot(features, self.parameters))
        return (predicted_probs >= threshold).astype(int)

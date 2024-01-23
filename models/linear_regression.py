import numpy as np
import matplotlib.pyplot as plt


class CustomLinearRegression:
    def __init__(self, learning_rate=0.0001, num_iterations=10000, cost_function='mse', learning_function='gradient_descent'):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.cost_function = cost_function
        self.learning_function = learning_function
        self.theta = None 

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        y = y

        self.theta = np.ones(X_b.shape[1]).reshape(-1, 1)

        if self.learning_function == 'ordinary_least_squares':
            self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            print(self.theta)
            return
        
        for iteration in range(self.num_iterations):
            # Obliczenie predykcji Macierz wektorów atrybutów * wagi
            y_pred = X_b.dot(self.theta)

            if self.learning_function == 'gradient_descent':
                if self.cost_function == 'mse':
                    gradient = self.base_gradient_mse(X_b, y, y_pred)
                elif self.cost_function == 'mae':
                    gradient = self.base_gradient_mae(X_b, y, y_pred)
                elif self.cost_function == 'huber_loss':
                    gradient = self.base_gradient_huber(X_b, y, y_pred)
                    
                self.theta -= self.learning_rate * gradient
                print(self.theta)

            if self.learning_function == 'stochastic_gradient_descent':
                elements = list(range(X.shape[0]))
                indices = np.random.choice(elements, round(len(elements)/5), replace=False)
                np.random.shuffle(indices)

                for idx in indices:
                    x_sample = X_b[idx:idx+1]
                    y_sample = y[idx:idx+1]

                    y_pred = x_sample.dot(self.theta)

                    if self.cost_function == 'mse':
                        gradient = self.base_gradient_mse(x_sample, y_sample, y_pred)
                    elif self.cost_function == 'mae':
                        gradient = self.base_gradient_mae(x_sample, y_sample, y_pred)
                    elif self.cost_function == 'huber_loss':
                        gradient = self.base_gradient_huber(x_sample, y_sample, y_pred)

                    self.theta -= self.learning_rate * gradient
                print(self.theta)

    def base_gradient_mse(self, X, y, y_pred):
        gradient = X.T.dot(2*(y_pred - y)) / X.shape[0]
        return gradient

    def base_gradient_mae(self, X, y, y_pred):
        errors = np.sign(y_pred - y)
        gradient = (np.sum(X * errors.reshape(X.shape[0], 1) / X.shape[0], axis=0)).reshape(-1, 1)
        return gradient

    def base_gradient_huber(self, X, y_pred, y):
        error = y_pred - y
        gradient = -X.T.dot(self.huber_loss_derivative(error)) / X.shape[0]
        print(gradient.shape)
        return gradient

    def huber_loss_derivative(self, error):
        delta = self.delta
        derivative = np.where(np.abs(error) <= delta, error, delta * np.sign(error))
        return derivative

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)

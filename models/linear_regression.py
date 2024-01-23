import numpy as np
import matplotlib.pyplot as plt
import gc

class CustomLinearRegression:
    def __init__(self, learning_rate=0.0002, num_iterations=10000, cost_function='mse', learning_function='gradient_descent'):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.cost_function = cost_function
        self.learning_function = learning_function
        self.y = None
        self.y_pred = None
        self.X_b = None
        self.theta = None  # Parametry modelu

    def fit(self, X, y):
        # Dodanie kolumny jednostkowej do zbioru danych trenujących z wartością stałą (bias)
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        
        self.X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.y = y

        # Inicjalizacja parametrów theta
        self.theta = np.ones(self.X_b.shape[1]).reshape(-1, 1)
  
     
        for iteration in range(self.num_iterations):
            # Obliczenie predykcji Macierz wektorów atrybutów * wagi
            self.y_pred = self.X_b.dot(self.theta)
            print(self.y_pred.shape)

            if self.learning_function == 'gradient_descent' and self.cost_function == 'mse':
                gradient = self.base_gradient_mse()
                self.theta -= self.learning_rate * gradient
                print(self.theta)
   
            if self.learning_function == 'stohastic_gradient_descent' and self.cost_function == 'mse':
                indices = list(range(X.shape[0]))
                np.random.shuffle(indices)

                for idx in indices:
                    x_sample = self.X_b[idx:idx+1]
                    y_sample = self.y[idx:idx+1]

                    self.y_pred = x_sample.dot(self.theta)
                    gradient = self.base_gradient_mse(x_sample, y_sample)
                    self.theta -= self.learning_rate * gradient


    def base_gradient_mse(self):
        gradient = self.X_b.T.dot(2*(self.y_pred - self.y)) / self.X_b.shape[0]
        return gradient


    def calculate_mse(self, X, y, y_pred):
        m = X.shape[0]
        mse = ((1/m)) * np.sum(np.power(y-y_pred, 2))
        return mse

    def calculate_mae(self, X, y, y_pred):
        m = X.shape[0]
        mae = (1/m) * np.sum(np.abs(y-y_pred))
        return mae

    def calculate_huber_loss(self, X, y, y_pred, delta=1.0):
        absolute_errors = np.abs(y - y_pred)
        square_errors = 0.5 * (absolute_errors ** 2)
        linear_errors = delta * (absolute_errors - 0.5 * delta)

        loss = np.where(absolute_errors <= delta, square_errors, linear_errors)
        return np.mean(loss)

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)



import numpy as np
import matplotlib.pyplot as plt
import constants

np.random.seed = constants.SEED


class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, cost_function='mse', learning_function='gradient_descent'):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.cost_function = cost_function
        self.learning_function = learning_function
        self.theta = None  # Parametry modelu

    def fit(self, X, y):
        # Dodanie kolumny jednostkowej do zbioru danych trenujących z wartością stałą (bias)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]

        # Inicjalizacja parametrów theta
        self.theta = np.random.randn(X_b.shape[1], 1)

        for iteration in range(self.num_iterations):
            # Obliczenie predykcji
            y_pred = X_b.dot(self.theta)

            # Obliczenie gradientu funkcji kosztu
            gradient = self.calculate_gradient(X_b, y, y_pred)

            print(f'theta: {self.theta}')
            print(f'theta_shape: {np.shape(self.theta)}')
            
            # Aktualizacja parametrów theta w zależności od metody optymalizacji
            if self.learning_function == 'gradient_descent':
                self.theta -= self.learning_rate * gradient
            
            # elif self.learning_function == 'stochastic_gradient_descent':
            #     random_index = np.random.randint(X_b.shape[0])
            #     xi = X_b[random_index:random_index+1]
            #     yi = y[random_index:random_index+1]
            #     gradient = self.calculate_gradient(xi, yi, xi.dot(self.theta))
            #     self.theta -= self.learning_rate * gradient
            # else:
            #     raise ValueError("Nieznana metoda optymalizacji.")

    def calculate_gradient(self, X, y, y_pred):
        # Obliczenie gradientu funkcji kosztu w zależności od funkcji kosztu
        if self.cost_function == 'mse':
            print(f'XT {X.T}')
            gradient = 2/X.shape[0] * X.T.dot(y_pred - y)
            print(f'gradient: {gradient}')
            return gradient

    def predict(self, X):
        # Dodanie kolumny jednostkowej do X (bias)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)


X = 2 * np.random.rand(10, 4)
y = 4 + 3 * X + np.random.randn(10, 1)

# model = CustomLinearRegression(learning_rate=0.01, num_iterations=1000, cost_function='mse', learning_function='gradient_descent')
# model.fit(X, y)


X = np.asarray([[2, 3, 4], [3, 3, 1], [1, 1, 1]])
y_pred = np.asarray([1, 2, 3])
y = np.asarray([1, 1, 1])
theta = np.asarray([1, 1, 3])

dot = X.dot(theta)

print(y_pred - y)
print(X.T)

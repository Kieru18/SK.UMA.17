SEED = 42

COST_FUNCTIONS = ['mse', 'mae', 'huber']
LEARNING_FUNCTIONS = ['gradient_descent', 'stochastic_gradient_descent', 'least_squares']

FEATURES_DIAMONDS = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
TARGET_DIAMONDS = 'price'

FEATURES_WINES = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
                  'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
TARGET_WINES = 'quality'

METRICS = ['mse', 'mae', 'r2']

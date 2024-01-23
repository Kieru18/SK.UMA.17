SEED = 42

COST_FUNCTIONS = ['mse', 'mae', 'huber']
LEARNING_FUNCTIONS = ['gradient_descent', 'stochastic_gradient_descent', 'least_squares']

FEATURES_DIAMONDS = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
TARGET_DIAMONDS = 'price'

FEATURES_WINES = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
                  'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
TARGET_WINES = 'quality'

FEATURES_HOUSING = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 
                    'total_bedrooms', 'population', 'households', 'median_income', 
                    'ocean_proximity']

TARGET_HOUSING = 'median_house_value'

METRICS = ['mse', 'mae', 'r2']

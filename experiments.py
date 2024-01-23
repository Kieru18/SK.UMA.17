import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict

from CONST import (SEED, COST_FUNCTIONS, LEARNING_FUNCTIONS, METRICS,
                   FEATURES_DIAMONDS, TARGET_DIAMONDS, FEATURES_WINES, TARGET_WINES,
                   FEATURES_HOUSING, TARGET_HOUSING)
from Dataset import Dataset
from models.RegressionModel import RegressionModel
from load_datasets import get_diamonds, get_wines, get_housing
from utils import train, calculate_metrics, scatter_plot, show_metrics, save_metrics
from models.linear_regression import CustomLinearRegression
# from models.random_forest_regression import CustomRandomForest

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


np.random.seed(SEED)
models: Dict[str, RegressionModel] = dict()


# load datasets
diamonds_df = get_diamonds()
wines_df = get_wines()
housing_df = get_housing()

diamonds = Dataset(diamonds_df, FEATURES_DIAMONDS, TARGET_DIAMONDS)
wines = Dataset(wines_df, FEATURES_WINES, TARGET_WINES)
housing = Dataset(housing_df, FEATURES_HOUSING, TARGET_HOUSING)

datasets: Dict[str, Dataset] = dict()
datasets['diamonds'] = diamonds
datasets['wines'] = wines
datasets['housing'] = housing

for dataset_name, dataset in datasets.items():
    dataset.split()

# own implementations of regression models
    
# for cost_function in COST_FUNCTIONS:
#     for learning_function in LEARNING_FUNCTIONS:
#         # models[f"LinearRegression_{cost_function}_{learning_function}"] = \
#         #     CustomLinearRegression(cost_function=cost_function, learning_function=learning_function)

#         # models[f"RandomForestRegressor_{cost_function}_{learning_function}"] = RandomForestRegressor()

#         pass

# ready to use models from sklearn
models["LinearRegression_sklearn_diamonds"] = LinearRegression()
models["LinearRegression_sklearn_wines"] = LinearRegression()
models["RandomForestRegressor_sklearn_diamonds"] = RandomForestRegressor(random_state=SEED)
models["RandomForestRegressor_sklearn_wines"] = RandomForestRegressor(random_state=SEED)

models["LinearRegression_mse_base_grad_wines"] = CustomLinearRegression(learning_rate=0.00001, num_iterations=1000, cost_function='mse', learning_function='gradient_descent')
models["LinearRegression_mse_base_grad_diamonds"] = CustomLinearRegression(learning_rate=0.00002, num_iterations=10000, cost_function='mse', learning_function='gradient_descent')
models["LinearRegression_mae_base_grad_wines"] = CustomLinearRegression(learning_rate=0.5, num_iterations=10000, cost_function='mae', learning_function='gradient_descent')
models["LinearRegression_mae_base_grad_diamonds"] = CustomLinearRegression(learning_rate=0.1, num_iterations=12000, cost_function='mae', learning_function='gradient_descent')
models["LinearRegression_huber_base_grad_wines"] = CustomLinearRegression(learning_rate=0.5, num_iterations=10000, cost_function='huber_loss', learning_function='gradient_descent')
models["LinearRegression_huber_base_grad_diamonds"] = CustomLinearRegression(learning_rate=0.1, num_iterations=12000, cost_function='huber_loss', learning_function='gradient_descent')

models["LinearRegression_mse_stohastic_grad_wines"] = CustomLinearRegression(learning_rate=0.00001, num_iterations=800, cost_function='mse', learning_function='stohastic_gradient_descent')
models["LinearRegression_mse_stohastic_grad_diamonds"] = CustomLinearRegression(learning_rate=0.00002, num_iterations=1400, cost_function='mse', learning_function='stohastic_gradient_descent')
models["LinearRegression_mae_stohastic_grad_wines"] = CustomLinearRegression(learning_rate=0.1, num_iterations=500, cost_function='mae', learning_function='stohastic_gradient_descent')
models["LinearRegression_mae_stohastic_grad_diamonds"] = CustomLinearRegression(learning_rate=0.1, num_iterations=500, cost_function='mae', learning_function='stohastic_gradient_descent')
models["LinearRegression_huber_stohastic_grad_wines"] = CustomLinearRegression(learning_rate=0.1, num_iterations=500, cost_function='huber_loss', learning_function='stohastic_gradient_descent')
models["LinearRegression_huber_stohastic_grad_diamonds"] = CustomLinearRegression(learning_rate=0.1, num_iterations=500, cost_function='huber_loss', learning_function='stohastic_gradient_descent')

# models["LinearRegression_ordinary_least_squares_diamonds"] = CustomLinearRegression(learning_function='ordinary_least_squares')
# models["LinearRegression_ordinary_least_squares_wines"] = CustomLinearRegression(learning_function='ordinary_least_squares')


for model_name, model in models.items():
        dataset_name = model_name.split('_')[-1]
        dataset = datasets[dataset_name]
        train(model, dataset)
        predictions = model.predict(dataset.x_test)
        scatter_plot(dataset.y_test, predictions, model_name, save=True)
        save_metrics(calculate_metrics(dataset.y_test, predictions), model_name)

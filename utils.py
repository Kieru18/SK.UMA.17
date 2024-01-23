import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict

from CONST import METRICS
from Dataset import Dataset
from models.RegressionModel import RegressionModel
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from typing import Dict



def train(model: RegressionModel, dataset: Dataset):
    model.fit(dataset.x_train, dataset.y_train)

def calculate_metrics(actual, predictions) -> Dict[str, float]:
    metrics: Dict[str, float] = dict()
    
    for metric in METRICS:
        if metric == 'mse':
            metrics[metric] = mean_squared_error(actual, predictions)
        elif metric == 'mae':
            metrics[metric] = mean_absolute_error(actual, predictions)
        elif metric == 'r2':
            metrics[metric] = r2_score(actual, predictions)
    return metrics

def scatter_plot(actual: pd.DataFrame, prediction: pd.DataFrame, 
                 model_name: str='', save: bool=False):
    plt.figure(figsize=(8, 8))
    plt.scatter(actual, prediction, color='blue', alpha=0.7)
    plt.plot([min(actual), max(actual)], [min(actual), max(actual)], 
              color='red', linestyle='--', linewidth=2)
    plt.title(f'{model_name} Actual vs. Predicted Values ')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    if save:
        os.makedirs('plots', exist_ok=True) 
        plt.savefig(f'plots/{model_name}.png')
    else:
        plt.show()

def show_metrics(metrics: Dict[str, float]) -> None:
    for metric in METRICS:
        print(f'{metric}: {metrics[metric]}')

def save_metrics(metrics: Dict[str, float], model_name: str) -> None:
    filepath = f'metrics/{model_name}.json'
    os.makedirs('metrics', exist_ok=True)  
    with open(filepath, 'w') as file:
        json.dump(metrics, file)


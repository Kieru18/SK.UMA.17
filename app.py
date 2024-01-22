import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from load_datasets import get_diamonds, get_wines
from models.linear_regression import CustomLinearRegression
from sklearn.model_selection import train_test_split

from CONST import SEED, COST_FUNCTIONS, LEARNING_FUNCTIONS


def main():
    diamonds_df = get_diamonds()
    wines_df = get_wines()

    diamonds_train, diamonds_test = train_test_split(diamonds_df, test_size=0.2, random_state=SEED)
    wines_train, wines_test = train_test_split(wines_df, test_size=0.2, random_state=SEED)

    models = [CustomLinearRegression]  # Reference to the class without calling its constructor

    for cost_function in COST_FUNCTIONS:
        for learning_function in LEARNING_FUNCTIONS:
            for model in models:
                print(f'------------------------')
                model = model(cost_function=cost_function, learning_function=learning_function)  # Call the constructor when needed
            
                print(f'theta: {model.theta}')
                print(f'cost_function: {model.cost_function}')
                print(f'learning_function: {model.learning_function}')
                print(f'------------------------')



if __name__ == "__main__":
    main()
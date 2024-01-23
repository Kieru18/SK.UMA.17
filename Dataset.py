from sklearn.model_selection import train_test_split
import pandas as pd
from CONST import SEED
from typing import List


class Dataset:
    def __init__(self, 
                 dataframe: pd.DataFrame, 
                 features: List[str], 
                 target: List[str], 
                 test_size: int = 0.2, 
                 random_state: int = SEED):
        self.dataframe = dataframe
        self.features = features
        self.target = target
        self.random_state = random_state
        self.test_size = test_size
        
    def split(self):
        self.x_train, self.x_test, self.y_train, self.y_test = \
            train_test_split(self.dataframe[self.features], 
                             self.dataframe[self.target], 
                             test_size=self.test_size, 
                             random_state=self.random_state)

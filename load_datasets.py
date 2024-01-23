import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_wines() -> pd.DataFrame:
    wines_df = pd.read_csv('data/wine_quality/winequality-white.csv', delimiter=';')

    return wines_df


def get_diamonds() -> pd.DataFrame:
    diamonds_df = pd.read_csv('data/diamonds/diamonds.csv', index_col='id')

    label_encoder = LabelEncoder()
    diamonds_df['cut'] = label_encoder.fit_transform(diamonds_df['cut'])
    diamonds_df['color'] = label_encoder.fit_transform(diamonds_df['color'])
    diamonds_df['clarity'] = label_encoder.fit_transform(diamonds_df['clarity'])

    return diamonds_df


def get_housing() -> pd.DataFrame:
    housing_df = pd.read_csv('data/housing/housing.csv')

    label_encoder = LabelEncoder()
    housing_df['ocean_proximity'] = label_encoder.fit_transform(housing_df['ocean_proximity'])

    return housing_df

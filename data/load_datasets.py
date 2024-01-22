import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def get_wines():
    wines_df = pd.read_csv('wine_quality/winequality-white.csv', delimiter=';')

    return wines_df


def get_diamonds():
    diamonds_df = pd.read_csv('diamonds/diamonds.csv', index_col='id')

    # Apply label encoding to non-numercial attributes
    label_encoder = LabelEncoder()
    diamonds_df['cut'] = label_encoder.fit_transform(diamonds_df['cut'])
    diamonds_df['color'] = label_encoder.fit_transform(diamonds_df['color'])
    diamonds_df['clarity'] = label_encoder.fit_transform(diamonds_df['clarity'])

    return diamonds_df

import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder


def main():
    diamonds = get_diamonds()
    white_wine = get_wines()

    print(diamonds.info())
    print(diamonds.head())
    print('--------------------------')
    print(white_wine.info())
    print(white_wine.head())


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


def generate_reports(diamonds, white_wine):

    profile_diamonds = ProfileReport(diamonds, title="Profile Report - Diamonds",
                                     html={'style': {'full_width': True}})

    profile_wine = ProfileReport(white_wine, title="Profile Report - White Wine",
                                 html={'style': {'full_width': True}})

    profile_diamonds.to_file(output_file="diamonds_report.html")
    profile_wine.to_file(output_file="wine_report.html")


if __name__ == '__main__':
    main()

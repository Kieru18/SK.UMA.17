import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
from load_datasets import get_diamonds, get_wines, get_housing


def main():
    diamonds = get_diamonds()
    white_wine = get_wines()
    housing = get_housing()

    print(diamonds.info())
    print(diamonds.head())
    print('--------------------------')
    print(white_wine.info())
    print(white_wine.head())
    print('--------------------------')
    print(housing.info())
    print(housing.head())


def generate_reports(diamonds, white_wine, housing):

    profile_diamonds = ProfileReport(diamonds, title="Profile Report - Diamonds",
                                     html={'style': {'full_width': True}})

    profile_wine = ProfileReport(white_wine, title="Profile Report - White Wine",
                                 html={'style': {'full_width': True}})
    
    profile_housing = ProfileReport(housing, title="Profile Report - Housing",
                                    html={'style': {'full_width': True}})

    profile_diamonds.to_file(output_file="diamonds_report.html")
    profile_wine.to_file(output_file="wine_report.html")
    profile_housing.to_file(output_file="housing_report.html")


if __name__ == '__main__':
    main()

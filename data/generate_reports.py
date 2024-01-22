import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport
from sklearn.preprocessing import LabelEncoder
from load_datasets import get_diamonds, get_wines


def main():
    diamonds = get_diamonds()
    white_wine = get_wines()

    print(diamonds.info())
    print(diamonds.head())
    print('--------------------------')
    print(white_wine.info())
    print(white_wine.head())


def generate_reports(diamonds, white_wine):

    profile_diamonds = ProfileReport(diamonds, title="Profile Report - Diamonds",
                                     html={'style': {'full_width': True}})

    profile_wine = ProfileReport(white_wine, title="Profile Report - White Wine",
                                 html={'style': {'full_width': True}})

    profile_diamonds.to_file(output_file="diamonds_report.html")
    profile_wine.to_file(output_file="wine_report.html")


if __name__ == '__main__':
    main()

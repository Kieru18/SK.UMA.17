import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data.preapre_datasets import get_diamonds, get_wines


def main():
    diamonds_df = get_diamonds()
    wines_df = get_wines()

    diamonds_df.info()
    wines_df.info()


if __name__ == "__main__":
    main()
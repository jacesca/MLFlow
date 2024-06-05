import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
from dotenv import load_dotenv
from functools import partial

from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             root_mean_squared_error, r2_score)

load_dotenv()
SEED = int(os.getenv("SEED"))
CRED = '\033[42m'
CEND = '\033[0m'


def prepare_environment():
    # Preparing the environment
    pd.set_option('display.max_columns', None)
    plt.rcParams.update({'axes.labelsize': 8, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'legend.fontsize': 8,
                         'axes.titlesize': 10, 'axes.titleweight': 'bold',
                         'font.size': 8, 'figure.titlesize': 10,
                         'figure.titleweight': 'bold'})
    np.random.seed(SEED)


def hprint(msg):
    print(CRED + msg + CEND)


def calculate_regressionmetrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"""Evaluating the model:
    MSE : {mse}
    RMSE: {rmse}
    MAE : {mae}
    R²  : {r2}
    """)


print = partial(print, sep='\n', end='\n\n')

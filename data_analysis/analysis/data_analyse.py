import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import mplfinance as mpf
import warnings
import os

# Ignore all warnings
warnings.filterwarnings("ignore")

# Step 2: Load the dataset
# 构建 CSV 文件的相对路径
csv_file_path = os.path.join('analysis', 'static', 'tesla_stock_data.csv')

# 使用相对路径打开 CSV 文件
#tesla_data = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True)


def Description(tesla_data):
    return tesla_data.describe().T

def MissingValueChecking(tesla_data):
    return tesla_data.isnull().sum()

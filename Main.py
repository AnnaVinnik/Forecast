import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
from fredapi import Fred
from statsmodels.tsa.arima.model import ARIMA

load_dotenv()

api_key = os.getenv('FRED_API_KEY')

np.random.seed(42)
length = 100
trend_arr = np.linspace(2, 5, length)
noise_arr = np.linspace(0, 0.5, length)
data_arr = trend_arr + noise_arr
dates = pd.date_range(start='2018-01-01', periods=length, freq='M')

fred = Fred(api_key=api_key)
print("API KEY:", os.getenv("FRED_API_KEY"))
data = fred.get_series('CLVMNACSCAB1GQHU')
print(data.tail())
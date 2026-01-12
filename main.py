import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import yfinance as yf

def get_data(stocks, start, end):
    data = yf.download(stocks, start=start, end=end, progress=False)['Close']
    returns = data.pct_change().dropna()
    mean = returns.mean()
    cov_matrix = returns.cov()
    return mean, cov_matrix

stocks = ['AAPL', 'GOOG', 'IBM', 'MSFT', 'AMZN']

end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=300)

mean, cov_matrix = get_data(stocks, start_date, end_date)

print(mean)

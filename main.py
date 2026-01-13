import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import datetime as dt
import yfinance as yf
from sklearn.covariance import LedoitWolf
from scipy.stats import t

def get_log_returns(stocks, start, end):
    prices = yf.download(stocks, start=start, end=end, progress=False)['Close']
    return np.log(prices / prices.shift(1)).dropna()

stocks = ['AAPL', 'GOOG', 'IBM', 'MSFT', 'AMZN']
end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=300)

log_returns = get_log_returns(stocks, start_date, end_date)

mu = log_returns.mean().values
lw = LedoitWolf()
cov = lw.fit(log_returns.values).covariance_


T = 100        # days
n_paths = 50
initial_value = 10_000
df = 5         # Student-t fat tails

weights = np.random.random(len(mu))
weights /= weights.sum()

L = np.linalg.cholesky(cov)

portfolio_values = np.full((T, n_paths), initial_value)

Z = t.rvs(df=df, size=(T, n_paths, len(mu)))


fig, ax = plt.subplots()
lines = [ax.plot([], [], lw=1)[0] for _ in range(n_paths)]

ax.set_xlim(0, T)
ax.set_ylim(initial_value * 0.7, initial_value * 1.5)
ax.set_xlabel("Days")
ax.set_ylabel("Portfolio Value")
ax.set_title("Dynamic Monte Carlo Portfolio Simulation")

# animation
def init():
    for line in lines:
        line.set_data([], [])
    return lines

def update(t):
    if t == 0:
        return lines

    correlated_returns = Z[t] @ L.T + mu
    portfolio_log_returns = correlated_returns @ weights

    portfolio_values[t] = portfolio_values[t-1] * np.exp(portfolio_log_returns)

    for i, line in enumerate(lines):
        line.set_data(range(t + 1), portfolio_values[:t + 1, i])

    return lines

ani = animation.FuncAnimation(
    fig,
    update,
    frames=T,
    init_func=init,
    blit=True,
    interval=50
)

plt.show()

import cvxpy as cp
import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt
from order_book import OrderBookTorch

def get_data(s):
    filename = f"{s}_order_book.bin"
    with open(filename, 'rb') as file:
        data = file.read()
        book = OrderBookTorch.deserialize(data)

    prices = []
    volumes = []
    timestamps = []

    for update_id, frame in book.frames.items():
        bid_prices = frame['bids'][:, 0].cpu().numpy()
        bid_volumes = frame['bids'][:, 1].cpu().numpy()
        ask_prices = frame['asks'][:, 0].cpu().numpy()
        ask_volumes = frame['asks'][:, 1].cpu().numpy()

        weighted_bid_price = np.sum(bid_prices * bid_volumes) / np.sum(bid_volumes) if np.sum(bid_volumes) > 0 else 0
        weighted_ask_price = np.sum(ask_prices * ask_volumes) / np.sum(ask_volumes) if np.sum(ask_volumes) > 0 else 0

        mid_weighted_price = (weighted_bid_price + weighted_ask_price) / 2
        total_volume = np.sum(bid_volumes) + np.sum(ask_volumes)

        prices.append(mid_weighted_price)
        volumes.append(total_volume)
        timestamps.append(int(frame["time"]))

    return pd.DataFrame({'price': prices[1:], 'volume': volumes[1:]}, index=pd.to_datetime(timestamps[1:], unit='ms'))

def resample_data(dfs, interval='1T'):
    resampled_dfs = {}
    for symbol, df in dfs.items():
        resampled_df = df.resample(interval).mean().interpolate(method='linear')
        resampled_dfs[symbol] = resampled_df
    return resampled_dfs

def prepare_data(symbols, interval):
    data = {}
    for symbol in symbols:
        data[symbol] = get_data(symbol)
    
    resampled_data = resample_data(data, interval)
    prices = pd.DataFrame({symbol: 100* df['price']/df['price'].iloc[0] for symbol, df in resampled_data.items()})
    return prices.bfill().ffill()

def forecast_future_returns(prices, forecast_horizon):
    log_returns = np.log(prices / prices.shift(1)).dropna()
    means = log_returns.mean()
    cov = log_returns.cov()
    forecasted_returns = np.random.multivariate_normal(means, cov, size=forecast_horizon)
    forecasted_returns_df = pd.DataFrame(forecasted_returns, columns=prices.columns)
    return forecasted_returns_df

def multi_period_optimization(current_weights, forecasted_returns, target_return, trading_cost_func, holding_cost_func):
    n_assets = forecasted_returns.shape[1]
    forecast_horizon = forecasted_returns.shape[0]

    # Variables
    w = cp.Variable((forecast_horizon, n_assets))  # weights for each period
    z = cp.Variable((forecast_horizon, n_assets))  # trades for each period

    # Parameters
    mu = forecasted_returns.values
    gamma = 1  # Risk aversion parameter

    # Objective
    objective = 0
    for t in range(forecast_horizon):
        ret = mu[t] @ w[t]
        risk = cp.quad_form(w[t], np.cov(forecasted_returns.T))
        trading_cost = trading_cost_func(z[t])
        holding_cost = holding_cost_func(w[t])
        objective += ret - gamma * risk - trading_cost - holding_cost

    # Constraints
    constraints = [cp.sum(w[t]) == 1 for t in range(forecast_horizon)]
    constraints += [w[t] == w[t-1] + z[t] for t in range(1, forecast_horizon)]
    constraints += [w[0] == current_weights]
    #constraints += [cp.sum(mu[t] @ w[t]) >= target_return for t in range(forecast_horizon)]
    constraints += [w >= 0]  # Non-negative weights
    constraints += [w <= 1]  # Upper bound on weights

    # Problem
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve()

    if w.value is None:
        print("Optimization problem did not solve successfully.")
        print("Status:", prob.status)
        return None, None

    return w.value, z.value

# Define trading and holding cost functions
def trading_cost(z):
    return cp.norm(z, 1)  # L1 norm as an example trading cost

def holding_cost(w):
    return cp.norm(w, 2)  # L2 norm as an example holding cost

class Allocator():
    def __init__(self, train_data):
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()
        self.current_weights = np.zeros(train_data.shape[1])

    def allocate_portfolio(self, asset_prices, lookback, forecast_horizon=50):
        self.running_price_paths = self.running_price_paths._append(asset_prices, ignore_index=True)

        future_returns = forecast_future_returns(self.running_price_paths.iloc[-lookback:], forecast_horizon)
        best_portfolio = self.optimize_portfolio(future_returns)
        if best_portfolio is None:
            print("Returning current weights due to optimization failure.")
            return self.current_weights

        self.current_weights = best_portfolio[0]  # Update current weights with the first period's optimal weights
        return self.current_weights

    def optimize_portfolio(self, forecasted_returns):
        max_sharpe = 0
        best_portfolio = None
        for target_return in np.linspace(0.000001, 0.0010, 500):  # Adjust range as needed
            optimal_weights, _ = multi_period_optimization(self.current_weights, forecasted_returns, target_return, trading_cost, holding_cost)
            if optimal_weights is None:
                continue
            optimal_return = np.mean(optimal_weights)
            optimal_risk = np.std(optimal_weights)
            if optimal_risk == 0:
                sharpe = 0
            else:
                sharpe = optimal_return / optimal_risk

            if sharpe > max_sharpe:
                max_sharpe = sharpe
                best_portfolio = optimal_weights

        return best_portfolio

def grading(train_data, test_data, lookback):
    weights = np.full(shape=(len(test_data.index), 5), fill_value=0.0)
    alloc = Allocator(train_data)
    for i in range(len(test_data)):
        weights[i, :] = alloc.allocate_portfolio(test_data.iloc[i, :], lookback)
        if np.sum(weights < -1) or np.sum(weights > 1):
            raise Exception("Weights Outside of Bounds")

    capital = [1]
    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i, :])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i, :]))
        net_change = np.dot(shares, np.array(test_data.iloc[i+1, :]))
        capital.append(balance + net_change)

    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]

    if np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0
    return sharpe, capital, weights

symbols = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "BNBUSDT", "SOLUSDT"]
interval = '1S'
prices = prepare_data(symbols, interval)

TRAIN = prices.iloc[:50]
TEST = prices.iloc[50:]
sharpe, capital, weights = grading(TRAIN, TEST, lookback=50)

print(sharpe, capital, weights)

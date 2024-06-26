import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from order_book import OrderBookTorch
from pypfopt import EfficientFrontier, risk_models, expected_returns

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

"""
def optimize_portfolio(prices):
    mu = expected_returns.mean_historical_return(prices=prices, frequency=6.5*252*3600)
    S = risk_models.sample_cov(prices)

    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    return cleaned_weights, ef.portfolio_performance(verbose=True)

"""
'''
    mu = expected_returns.ema_historical_return(prices, span=50)
    S = risk_models.exp_cov(prices, span=50)

    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe(risk_free_rate=0)
    cleaned_weights = ef.clean_weights()

    return cleaned_weights, ef.portfolio_performance(verbose=True)
    
'''

import numpy as np
import pandas as pd
from scipy.optimize import minimize

def target_return_constraint(target_return, mean_returns):
    return lambda weights: target_return - np.dot(weights, mean_returns)

def optimal_portfolio_scipy(returns, target_return):
    # Calculate expected returns and the covariance matrix
    mean_returns = np.mean(returns, axis=1)
    cov_matrix = np.cov(returns)
    #print("cov_matrix", cov_matrix)

    # Number of assets
    num_assets = len(mean_returns)

    # Objective function (minimize portfolio variance)
    def portfolio_variance(weights):
        return weights.T @ cov_matrix @ weights

    # Constraints (weights sum to 1 and meet target return)
    constraints = [
        {'type': 'eq', 'fun': target_return_constraint(target_return, mean_returns)}]

    # Bounds (allow short selling, or adjust as needed)
    bounds = tuple((-1, 1) for _ in range(num_assets))

    # Initial guess (equal weighting)
    initial_guess = np.array([1./num_assets] * num_assets)

    # Portfolio optimization
    result = minimize(portfolio_variance, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    # Check if the optimization was successful
    if not result.success:
        raise ValueError("Optimization failed")

    # Optimal weights
    optimal_weights = result.x

    # Expected return and risk of the optimal portfolio
    optimal_return = np.dot(optimal_weights, mean_returns)
    optimal_risk = np.sqrt(result.fun)

    sharpe = (optimal_return) / optimal_risk
    return optimal_weights, optimal_return, optimal_risk, sharpe


def get_optimum_allocation(return_vec):
    max_sharpe = 0
    best_portfolio=False
    for tr in np.linspace(0.000001, 0.0010, 500):  # Adjust range as needed
        try:
            weights, ret, risk, sharpe = optimal_portfolio_scipy(return_vec, tr)
            if sharpe > max_sharpe:
                max_sharpe = sharpe
                best_portfolio = (weights, ret, risk, sharpe)
        except ValueError as e:
            print(e, tr)
            break  # or continue, depending on your preference for handling optimization failures
    return best_portfolio

symbols = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "BNBUSDT", "SOLUSDT"]
interval = '1S'  # Change the interval as needed (e.g., '5T' for 5 minutes, '1H' for 1 hour)
prices = prepare_data(symbols, interval)

print(prices)
#plt.plot(prices)
#plt.legend(list(prices.columns))
#plt.show()
#weights, performance = optimize_portfolio(prices)

#print("Optimal weights:", weights)
#print("Performance:", performance)

def forecast_future_returns(prices, forecast_horizon):
    log_returns = np.log(prices / prices.shift(1)).dropna()
    means = log_returns.mean()
    cov = log_returns.cov()
    forecasted_returns = np.random.multivariate_normal(means, cov, size=forecast_horizon)
    forecasted_returns_df = pd.DataFrame(forecasted_returns, columns=prices.columns)
    return forecasted_returns_df


class Allocator():
    def __init__(self, train_data):
        '''
        Anything data you want to store between days must be stored in a class field
        '''

        self.running_price_paths = train_data.copy()

        self.train_data = train_data.copy()


    def allocate_portfolio(self, asset_prices, lookback, forecast_horizon=50):
        self.running_price_paths = self.running_price_paths._append(asset_prices, ignore_index=True)

        future_returns = forecast_future_returns(self.running_price_paths.iloc[-lookback:], forecast_horizon)
        future_return_vec = np.array(future_returns.values).T
        best_portfolio = get_optimum_allocation(future_return_vec)
        return best_portfolio[0]

    """
    def allocate_portfolio(self, asset_prices, lookback):
        '''
        asset_prices: np array of length 6, prices of the 6 assets on a particular day
        weights: np array of length 6, portfolio allocation for the next day
        '''
        self.running_price_paths = self.running_price_paths._append(asset_prices, ignore_index = True)
        
        #plt.plot(self.running_price_paths.iloc[-lookback:])
        #plt.show()

        diffs = self.running_price_paths.pct_change().dropna()
        return_vec = np.array(diffs.values).T
        #weights, performance = optimize_portfolio(self.running_price_paths.iloc[-lookback:])

        best_portfolio = get_optimum_allocation(return_vec[:,-1*lookback:])
        #print(weights)
        #print(list(map(float, weights.values())))
        return best_portfolio[0]#list(map(float, weights.values()))
    """



def grading(train_data, test_data, lookback):
    '''
    Grading Script
    '''
    weights = np.full(shape=(len(test_data.index),5), fill_value=0.0)
    alloc = Allocator(train_data)
    for i in range(0,len(test_data)):
        print(i)
        weights[i,:] = alloc.allocate_portfolio(test_data.iloc[i,:], lookback)
        #print(weights[i,:])
        if np.sum(weights < -1) or np.sum(weights > 1):
            raise Exception("Weights Outside of Bounds")

    capital = [1]

    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i,:])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i,:]))
        net_change = np.dot(shares, np.array(test_data.iloc[i+1,:]))
        capital.append(balance + net_change)

    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]

    if np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0
    print(sharpe)
    return sharpe, capital, weights


TRAIN = prices.iloc[:50]
TEST = prices.iloc[50:]
sharpe, capital, weights= grading(TRAIN, TEST, lookback=50)

print(sharpe, capital, weights)


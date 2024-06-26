import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from order_book import OrderBookTorch
import cvxportfolio as cvx

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
    prices = pd.DataFrame({symbol: 100 * df['price'] / df['price'].iloc[0] for symbol, df in resampled_data.items()})
    return prices.bfill().ffill()

def forecast_future_returns(prices, forecast_horizon=100):
    log_returns = np.log(prices / prices.shift(1)).dropna()
    means = log_returns.mean()
    cov = log_returns.cov()
    forecasted_returns = np.random.multivariate_normal(means, cov, size=forecast_horizon)
    forecasted_returns_df = pd.DataFrame(forecasted_returns, columns=prices.columns)
    return forecasted_returns_df

symbols = ["BTCUSDT", "ETHUSDT", "LTCUSDT", "BNBUSDT", "SOLUSDT"]
interval = '1s'
prices = prepare_data(symbols, interval)

returns = prices.pct_change().dropna()
#market_data = cvx.UserProvidedMarketData(returns=returns, prices=prices)

returns['cash'] = 0


market_data = cvx.UserProvidedMarketData(returns=returns, prices=prices, cash_key='cash', min_history=pd.Timedelta('1 second'))


class Allocator():
    def __init__(self, market_data):
        self.market_data = market_data
        self.transaction_costs = {symbol: 0.001 for symbol in market_data.returns.columns}  # Example transaction cost parameter a

        #self.running_price_paths = train_data.copy()
        #self.train_data = train_data.copy()
        #self.transaction_costs = {symbol: 0.001 for symbol in train_data.columns}  # Example transaction cost parameter a

    def allocate_portfolio(self, lookback, forecast_horizon=100):
        #self.running_price_paths = self.running_price_paths._append(asset_prices, ignore_index=True)
        #future_returns = forecast_future_returns(self.running_price_paths.iloc[-lookback:], forecast_horizon)
        
        # Define the multiperiod optimizer
        #transaction_costs = cvx.TransactionCost(a=self.transaction_costs)
        #holding_costs = cvx.HoldingCost(short_fees=0.01, long_fees=0.01)  # Example holding cost parameters
        
        #risk_model = cvx.FullCovariance(self.running_price_paths.iloc[-lookback:].pct_change().cov().dropna())#cvx.forecast.HistoricalFactorizedCovariance().estimate(self.running_price_paths.iloc[-lookback:].pct_change().dropna()))
        #returns_forecast = cvx.ReturnsForecast(future_returns)
        
        lookback_prices = self.market_data.prices.iloc[-lookback:]
        future_returns = forecast_future_returns(lookback_prices, forecast_horizon)
        
        # Define the multiperiod optimizer
        transaction_costs = cvx.TransactionCost(a=self.transaction_costs)
        holding_costs = cvx.HoldingCost(short_fees=0.01, long_fees=0.01)  # Example holding cost parameters
        
        risk_model = cvx.FullCovariance()  # Using FullCovariance with default settings
        returns_forecast = cvx.ReturnsForecast(future_returns)

        gamma_risk, gamma_trade, gamma_hold = 5., 1., 1.
        policy = cvx.MultiPeriodOpt(
            objective=cvx.ReturnsForecast(future_returns) - gamma_risk * risk_model - gamma_trade * transaction_costs - gamma_hold * holding_costs,
            #costs=[transaction_costs, holding_costs],
            planning_horizon=100
        )
        
        #initial_portfolio = pd.Series(0.2, index=self.running_price_paths.columns)  # Example initial portfolio
        #weights = policy.execute(initial_portfolio, self.running_price_paths.iloc[-lookback:])
        
        initial_portfolio = pd.Series(0.2, index=self.market_data.returns.columns)  # Example initial portfolio
        weights = policy.execute(initial_portfolio, self.market_data)

        return weights
  

def grading(train_data, test_data, lookback, forecast_horizon=100):
    train_data.index = train_data.index.tz_localize('UTC')
    test_data.index = test_data.index.tz_localize('UTC')

    market_data_train = cvx.UserProvidedMarketData(returns=train_data.pct_change().dropna(), prices=train_data, cash_key='cash', min_history=pd.Timedelta('1 second'))
    market_data_test = cvx.UserProvidedMarketData(returns=test_data.pct_change().dropna(), prices=test_data, cash_key='cash', min_history=pd.Timedelta('1 second'))
    
    weights = np.full(shape=(len(test_data.index), len(test_data.columns)), fill_value=0.0)
    alloc = Allocator(market_data_train)
    for i in range(len(test_data)):
        print(i)
        weights[i, :] = alloc.allocate_portfolio(lookback, forecast_horizon)
        if np.sum(weights < -1) or np.sum(weights > 1):
            raise Exception("Weights Outside of Bounds")

    capital = [1]
    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i, :])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i, :]))
        net_change = np.dot(shares, np.array(test_data.iloc[i + 1, :]))
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
sharpe, capital, weights = grading(TRAIN, TEST, lookback=50)

print(sharpe, capital, weights)
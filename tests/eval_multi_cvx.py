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

def resample_data(dfs, interval='1s'):
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

plt.plot(prices)
plt.show()
returns = prices.pct_change().dropna()
#market_data = cvx.UserProvidedMarketData(returns=returns, prices=prices)

returns['cash'] = 0


market_data = cvx.UserProvidedMarketData(returns=returns, cash_key='cash', min_history=pd.Timedelta('1 second'))
print(market_data.returns)
'''
r_hat_with_cash = market_data.returns.rolling(
    window=50).mean().shift(1).dropna()
Sigma_hat_without_cash = market_data.returns.iloc[:, :-1
    ].rolling(window=50).cov().shift(4).dropna()

r_hat = r_hat_with_cash.iloc[:, :-1]
r_hat_cash = r_hat_with_cash.iloc[:, -1]
print('Expected returns forecast:')
print(r_hat_with_cash)

'''
HALF_SPREAD = 0.0001 / 2#10E-4

tcost_model = cvx.TcostModel(a=HALF_SPREAD, b=None)
hcost_model = cvx.HcostModel(short_fees=0)

#risk_model = cvx.FullSigma(Sigma_hat_without_cash)

leverage_limit = cvx.LeverageLimit(3)

gamma_risk, gamma_trade, gamma_hold = 5, 1., 1.

mpo_policy= cvx.MultiPeriodOptimization(cvx.ReturnsForecast(decay=0.9)
                                    - gamma_risk * cvx.FullSigma()
                                    - gamma_trade * tcost_model
                                    - gamma_hold * hcost_model,
                                    [leverage_limit],
                                    planning_horizon=6,
                                    )

market_sim = cvx.MarketSimulator(
    market_data = market_data,
    costs = [
        cvx.TcostModel(a=HALF_SPREAD, b=None),
        cvx.HcostModel(short_fees=0)])

# Initial portfolio, uniform on non-cash assets.
init_portfolio = pd.Series(
    index=market_data.returns.columns, data=200000.)
init_portfolio.cash = 0

print(init_portfolio)

#start_time = '2024-06-25 19:41:27'
#end_time = '2024-06-25 21:22:51'
start_time = '2024-06-24 19:23:25'
end_time = '2024-06-24 19:32:45'

results = market_sim.run_multiple_backtest(
    h=[init_portfolio]*2,
    start_time=start_time,
    end_time=end_time,
    policies=[mpo_policy, cvx.Hold()]
)

print('Back-test result, multi-period optimization policy:')
print(results[0])

print('Back-test result, Hold policy:')
print(results[1])

results[0].v.plot(label='MPO')
results[1].v.plot(label='Hold policy')
plt.title('Portfolio total value in time (USD)')
plt.legend()
plt.show()

results[0].w.plot()
plt.title('MPO weights in time')
plt.show()
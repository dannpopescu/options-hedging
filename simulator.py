import numpy as np
from scipy.stats import norm

from const import TRADING_DAYS_PER_YEAR
from const import STOCK, OPTONS, DELTA, TTM, DONE


class GBMSimulator:
    def __init__(self, init_price, mu, sigma, K, r, q, trading_freq, maturity):
        self.init_price = init_price
        self.mu = mu
        self.sigma = sigma
        self.K = K
        self.r = r
        self.q = q
        self.trading_freq = trading_freq
        self.maturity = maturity

        self.stock_prices = []
        self.options_values = []
        self.deltas = []
        self.ttm = []
        self.current_step = -1

    def days_to_maturity(self):
        return len(self.stock_prices) - 1

    def generate_new_path(self):
        self.current_step = 0

        self.stock_prices, \
        self.options_values, \
        self.deltas, \
        self.ttm = generate_stock_options_path(init_price=self.init_price,
                                               mu=self.mu,
                                               sigma=self.sigma,
                                               K=self.K,
                                               r=self.r,
                                               q=self.q,
                                               trading_freq=self.trading_freq,
                                               maturity=self.maturity)

    def next_state(self):
        """
        :return : stock_price, options_price, delta, ttm, done
        """
        if self.current_step == -1:
            raise RuntimeError("The simulator was not initialized!")

        s, v, d, t = self.stock_prices[self.current_step], \
                     self.options_values[self.current_step], \
                     self.deltas[self.current_step], \
                     self.ttm[self.current_step]

        self.current_step += 1
        done = self.current_step == len(self.stock_prices)

        return {
            STOCK: s,
            OPTONS: v,
            DELTA: d,
            TTM: t,
            DONE: done
        }


def generate_stock_options_path(init_price, mu, sigma, K, r, q, trading_freq, maturity):
    """ Simulate a path of stock prices using Geometric Brownian Motion and compute call option price
    at every time step.

    :param init_price: initial stock price
    :param mu: expected return per one period T
    :param sigma: volatility per one period T
    :param K: strike price
    :param r: risk-free rate
    :param q: dividend-yield rate
    :param trading_freq: the frequency of the generated stock prices (1=daily, 2=every two days, 0.5=twice per day)
    :param maturity: maturity of the options, in years

    :return stock_prices: a 2d array of generated stock prices (1 row = 1 path)
    :return call_values: a 2d array of call option prices for every stock price generated
    :return deltas: a 2d array of options' deltas
    """
    stock_prices, time_steps = generate_stock_price_path_gbm(init_price, mu, sigma, trading_freq, maturity)
    ttm = np.flip(time_steps)
    call_values, deltas = bsm_call(stock_prices, K, r, q, sigma, ttm)
    return stock_prices, call_values, deltas, ttm


def generate_stock_price_path_gbm(init_price, mu, sigma, frequency, time_horizon):
    """ Generate simulations of stock prices using Geometric Brownian Motion
    :param init_price: initial stock price
    :param mu: expected return per one period T
    :param sigma: volatility per one period T
    :param frequency: the frequency of the generated stock prices (1=daily, 2=every two days, 0.5=twice per day)
    :param time_horizon: the time horizon for which to generate prices, in years

    :return paths: a 2d array where every row is a simulated path
    :return time_steps: an array representing the points in time of the prices
    """
    dt = frequency / TRADING_DAYS_PER_YEAR
    n_steps = int(time_horizon / dt)
    steps = np.linspace(1, n_steps, n_steps)
    time_steps = dt * steps

    dW = np.random.normal(scale=np.sqrt(dt), size=n_steps)
    W = np.cumsum(dW)
    paths = init_price * np.exp((mu - 0.5 * sigma ** 2) * time_steps + sigma * W)

    paths = np.insert(paths, 0, init_price)
    time_steps = np.insert(time_steps, 0, 1e-15)

    return paths, time_steps


def bsm_call(paths, K, r, q, sigma, ttm):
    """ Price call options using a 2d array of price paths
    :param paths: stock price paths
    :param K: strike price
    :param r: risk-free rate
    :param q: dividend-yield rate
    :param sigma: volatility
    :param ttm: time-to-maturities (should have the same length as one path)
    """
    d1 = (np.log(paths / K) + (r - q + 0.5 * sigma ** 2) * ttm) / (sigma * np.sqrt(ttm))
    d2 = d1 - sigma * np.sqrt(ttm)
    deltas = norm.cdf(d1)
    call_values = paths * np.exp(-q * ttm) * deltas - K * np.exp(-r * ttm) * norm.cdf(d2)
    return call_values, deltas

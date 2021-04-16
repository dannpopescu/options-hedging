import gym
import numpy as np

from simulator import GBMSimulator
from const import STOCK, OPTONS, TTM, DONE, HOLDING, SHARES_PER_CONTRACT


class HedgingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, init_price, mu, sigma, strike_price, r, q, trading_freq, maturity, trading_cost, seed):
        """
        :param init_price: initial stock price
        :param mu: expected rate of return continuously compounded
        :param sigma: volatility rate
        :param strike_price: strike price
        :param T: option maturity in years
        :param k: transaction cost as percentage of trade value
        """
        super(HedgingEnv, self).__init__()

        self.simulator = GBMSimulator(init_price=init_price,
                                      mu=mu,
                                      sigma=sigma,
                                      K=strike_price,
                                      r=r,
                                      q=q,
                                      trading_freq=trading_freq,
                                      maturity=maturity,
                                      seed=seed)

        self.trading_cost = trading_cost
        self.strike_price = strike_price
        self.prev_state = []
        self.current_state = []

        self.action_space = gym.spaces.Box(0, SHARES_PER_CONTRACT, shape=(1,))

    def reset(self):
        self.simulator.generate_new_path()
        self.current_state = self.simulator.next_state()
        self.current_state[HOLDING] = 0
        return [self.current_state[HOLDING], self.current_state[STOCK], self.current_state[TTM]]

    def step(self, asset_to_be_held):
        """
        :param asset_to_be_held: amount of asset to be held for the next period
        :return: [observation, reward, done, info]
        """
        self.prev_state = self.current_state
        self.current_state = self.simulator.next_state()
        self.current_state[HOLDING] = asset_to_be_held

        observation = [self.current_state[HOLDING], self.current_state[STOCK], self.current_state[TTM]]
        reward = self.compute_reward()

        return [observation, reward, self.current_state[DONE], []]

    def compute_reward(self):
        options_value_change = (self.current_state[OPTONS] - self.prev_state[OPTONS]) * SHARES_PER_CONTRACT
        stock_position_change = self.current_state[HOLDING] * (self.current_state[STOCK] - self.prev_state[STOCK])
        transaction_cost = self.trading_cost * np.abs(self.prev_state[STOCK] * (self.current_state[HOLDING] - self.prev_state[HOLDING]))

        reward = -options_value_change + stock_position_change - transaction_cost

        if self.current_state[DONE]:
            reward -= self.trading_cost * np.abs(self.current_state[HOLDING] * self.current_state[STOCK])  # liquidate the hedge

        return -np.abs(reward)

    def render(self, mode='human', close=False):
        pass

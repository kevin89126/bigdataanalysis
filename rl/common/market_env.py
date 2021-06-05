# -*- coding: utf-8 -*-
import math
import os
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from random import uniform
from pandas import DataFrame


INIT_WEALTH = 1000000

def proration_weights(action):
    if action.sum() == 0:
        action = np.random.rand(*action.shape)
    return action / action.sum()


def simple_return_reward(env, **kwargs):
    reward = env.profit
    return reward


def sharpe_ratio_reward(env, **kwargs):
    profit_min = abs(np.array(env.profits).min())
    if env.profit <= 0:
        reward = env.profit * profit_min
    else:
        reward = env.profit / profit_min
    return reward


def risk_adjusted_reward(env, threshold: float=float("inf"), drop_only: bool = False):
    reward = env.profit
    if (abs(reward) < threshold):
        return reward
    if (reward >= 0 and drop_only):
        return reward
    reward = reward - (abs(reward) - threshold)

    return reward

def get_max_drawdown(env, df, period_start_date, current_date):
    res = []
    cur_df = df[period_start_date:current_date]
    if period_start_date == current_date:
        return 1
    for i in cur_df:
        # From max to find min
        _max = cur_df[i].max()
        _idx_max = cur_df[i].idxmax()
        _min = cur_df[i][_idx_max:].min()
        mdd_max_min = _max - _min
        
        # From min to find max
        _min = cur_df[i].min()
        _idx_min = cur_df[i].idxmin()
        _max = cur_df[i][:_idx_min].max()
        mdd_min_max = _max - _min
        mdd = max(mdd_max_min, mdd_min_max)
        
        res.append(mdd)
    return res


def resample_backfill(df, rule):
    return df.apply(lambda x: 1+x).resample(rule).backfill()


def resample_relative_changes(df, rule):
    return df.apply(lambda x: 1+x).resample(rule).prod().apply(lambda x: x-1)


class MarketEnv(gym.Env):

    def __init__(self, raw_data: DataFrame, returns: DataFrame, features: DataFrame, show_info=False, trade_freq='days',
                 action_to_weights_func=proration_weights,
                 reward_func=simple_return_reward,
                 reward_func_kwargs=dict(),
                 noise=0,
                 state_scale=1,
                 trade_pecentage=0.1):
        self._load_data(raw_data=raw_data, returns=returns, features=features, show_info=show_info, trade_freq=trade_freq)
        self._init_action_space()
        self._init_observation_space()
        self.trade_pecentage = trade_pecentage
        self.start_index, self.current_index, self.end_index = 0, 0, 0
        self.action_to_weights_func = action_to_weights_func
        self.reward_func = reward_func
        self.reward_func_kwargs = reward_func_kwargs
        self.noise = noise
        self.state_scale = state_scale
        self.seed()
        self.reset()

    def _load_data(self, raw_data: DataFrame, returns: DataFrame, features: DataFrame, show_info, trade_freq):
        #resample_rules = {
        #    'days': 'D',
        #    'weeks': 'W',
        #    'months': 'M'
        #}

        #if(trade_freq not in resample_rules.keys()):
        #    raise ValueError(f"trade_freq '{trade_freq}' not supported, must be one of {list(resample_rules.keys())}")

        #if returns.isnull().values.any():
        #    raise ValueError('At least one null value in investments')
        #if features.isnull().values.any():
        #    raise ValueError('At least one null value in investments')
        # make sure dataframes are sorted
        returns = returns.sort_index().sort_index(axis=1)
        features = features.sort_index().sort_index(axis=1)
        raw_data = raw_data.sort_index().sort_index(axis=1)

        #features = resample_backfill(features, resample_rules[trade_freq]).dropna()
        # TODO no need to change
        # Scale features to -1 and 1
        #for col in features.columns:
        #    mean = features[col].mean()
        #    std = features[col].std()
        #    features[col] = (features[col]-mean)/std
        # TODO no need to resample
        # resample based on trade frequency e.g. weeks or months
        #returns = resample_relative_changes(returns, resample_rules[trade_freq])

        # Only keep feature data within peroid of investments returns
        #features = features[(features.index.isin(returns.index))]
        # Only keep investment retuns with features
        #returns = returns[(returns.index.isin(features.index))]
        self.features = features
        self.returns = returns
        self.raw_data = raw_data
        if show_info:
            sd = returns.index[0]
            ed = returns.index[-1]
            print(f'Trading Frequency: {trade_freq}')
            print(f'{self.investments_count} investments loaded')
            print(f'{self.features_count} features loaded')
            print(f'Starts from {sd} to {ed}')

    def _init_action_space(self):
        action_space_size = self.investments_count
        self.min_action = np.zeros(action_space_size)
        self.max_action = np.ones(action_space_size)
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, dtype=np.float32)

    def _init_observation_space(self):
        observation_space_size = self.features_count
        self.low_state = np.full(observation_space_size, -1)
        self.high_state = np.ones(observation_space_size)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)

    @property
    def investments_count(self):
        return len(self.returns.columns)

    @property
    def features_count(self):
        return len(self.features.columns)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        #print(f'step {self.returns.index[self.current_index]}')
        if self.current_index > self.end_index:
            raise Exception(f'current_index {current_index} exceed end_index{self.end_index}')

        done = True if (self.current_index >= self.end_index) else False
        # update weight
        self.weights = self.action_to_weights_func(action)
        self.episode += 1
        self.current_index += 1
        # update investments and wealth
        previous_investments = self.investments

        # invest all wealth to target investments
        target_investments = self.weights

        # Investment
        invest_wealth = round(self.wealth * 0.8)

        # Get profit
        inv_return = self.returns.iloc[self.current_index]
        self.profits = np.multiply(self.weights, inv_return)
        # w_n = w_n-1 * (1+r)
        self.profit = self.profits.sum()

        # Get current wealth after investment
        self.wealth = self.wealth * (1 + self.profit)

        # Return reward
        reward = self.reward_func(self,**self.reward_func_kwargs)
        self.reward = reward

        info = self._get_info()
        state = self._get_state()
        return state, reward, done, info

    def render(self):
        pass

    def reset(self):
        total_index_count = len(self.returns.index)
        last_index = total_index_count-2
        #if (self.trade_pecentage >= 1):
        self.start_index = 0
        self.end_index = last_index
        #else:
        #    self.start_index = np.random.randint(low=0, high=last_index*(1-self.trade_pecentage))
        #    self.end_index = int(self.start_index + total_index_count*self.trade_pecentage)
        #    self.end_index = min(self.end_index, last_index)
        #print(f'total: {total_index_count}, start index: {self.start_index}, end index: {self.end_index}')
        self.current_index = self.start_index
        self.investments = np.zeros(self.investments_count)
        self.weights = np.zeros(self.investments_count)
        self.wealth = INIT_WEALTH
        self.max_weath = self.wealth
        self.max_drawdown = 0
        self.mean = 0
        self.mean_square = 0
        self.episode = 0
        self.profit = 0
        return self._get_state()

    def _get_state(self):
        # Use profix as random noise
        #noise = np.random.normal(0, abs(self.profit), self.observation_space.shape)

        noise = []
        state = self.features.iloc[self.current_index].to_numpy()*self.state_scale
        for s in state:
            n = np.random.normal(0, abs(s/10))
            noise.append(n)
 
        state = state + noise
        np.clip(state, -1, 1, out=state)
        if (state.shape != self.observation_space.shape):
            raise Exception('Shape of state {state.shape} is incorrect should be {self.observation_space.shape}')
        return state

    def _get_info(self):
        start_date = self.returns.index[self.start_index]
        current_date = self.returns.index[self.current_index]
        #trade_days = (current_date-start_date).days
        # TODO
        # Monthly profit use 12 instead of 365?
        cagr = math.pow(1 + self.profit, 12) - 1
        if (self.episode == 1):
            std = 0
        else:
            k = ((self.episode)/(self.episode-1))**0.5
            a = self.mean
            b = self.mean_square
            std = k*(b-a**2)**0.5

        info = {
        #    'trade_days': trade_days,
            'wealths': self.wealth,
        #    'max_weath': self.max_weath,
            'cagr': cagr,
        #    'std': std,
        #    'mean': self.mean,
        #    'mean_square': self.mean_square,
        #    'mdd': self.max_drawdown,
            'profit': self.profit,
            'reward': self.reward,
        #    'dd': self.drawdown,
        #    'episode': self.episode,
        }
        return info

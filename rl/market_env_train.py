import gym
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.launchers.launcher_util import setup_logger
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
import pandas as pd
from pandas import Timestamp
import os
import common
import matplotlib.pyplot as plt
from common.trainer import get_trainer
from common.market_env import simple_return_reward, sharpe_ratio_reward, risk_adjusted_reward
from common.matplotlib_extend import plot_ma
import rlkit.torch.pytorch_util as ptu
import numpy as np
from datetime import datetime
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
import gtimer as gt

from constants import TRAIN_DATA, VALIDATE_DATA, FEATURE_TRAIN_DATA, FEATURE_VALIDATE_DATA, RAW_TRAIN_DATA, \
    RAW_VALIDATE_DATA, TRAIN_LOG_PATH, EPOCH, FINAL_KPIS, MEAN_KPIS, ENABLE_EPOCH_PLOT

def load_dataset():
    # Get data path
    current_folder = os.path.dirname(__file__)
    ret_csv_train = os.path.join(current_folder, TRAIN_DATA)
    ret_csv_val = os.path.join(current_folder, VALIDATE_DATA)
    features_csv_train = os.path.join(current_folder, FEATURE_TRAIN_DATA)
    features_csv_val = os.path.join(current_folder, FEATURE_VALIDATE_DATA)
    raw_data_csv_train = os.path.join(current_folder, RAW_TRAIN_DATA)
    raw_data_csv_val = os.path.join(current_folder, RAW_VALIDATE_DATA)

    # Load data
    df_ret_train = pd.read_csv(ret_csv_train, parse_dates=['Date'], index_col=['Date'])
    df_ret_val = pd.read_csv(ret_csv_val, parse_dates=['Date'], index_col=['Date'])
    df_feature_train = pd.read_csv(features_csv_train, parse_dates=['Date'], index_col=['Date'])
    df_feature_val = pd.read_csv(features_csv_val, parse_dates=['Date'], index_col=['Date'])
    df_raw_data_train  = pd.read_csv(raw_data_csv_train, parse_dates=['Date'], index_col=['Date'])
    df_raw_data_val  = pd.read_csv(raw_data_csv_val, parse_dates=['Date'], index_col=['Date'])
    return df_ret_train, df_ret_val, df_feature_train, df_feature_val, df_raw_data_train, df_raw_data_val


gym.envs.register(id='MarketEnv-v0', entry_point='common.market_env:MarketEnv', max_episode_steps=1000)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = LOG_PATH.format(timestamp)

def train_model(variant):
    gt.reset_root() 
    setup_logger('name-of-experiment', variant=variant,
             snapshot_mode='gap_and_last', snapshot_gap=20, log_dir=log_dir)

    expl_env_kwargs = variant['expl_env_kwargs']
    eval_env_kwargs = variant['eval_env_kwargs']
    trainer_kwargs = variant['trainer_kwargs']

    df_ret_train, df_ret_val, df_feature_train, df_feature_val, df_raw_data_train, df_raw_data_val = load_dataset()
    df_ret_train.to_csv(os.path.join(log_dir, 'df_ret_train.csv'))
    df_ret_val.to_csv(os.path.join(log_dir, 'df_ret_val.csv'))
    df_feature_train.to_csv(os.path.join(log_dir, 'df_feature_train.csv'))
    df_feature_val.to_csv(os.path.join(log_dir, 'df_feature_val.csv'))
    df_raw_data_train.to_csv(os.path.join(log_dir, 'df_raw_data_train.csv'))
    df_raw_data_val.to_csv(os.path.join(log_dir, 'df_raw_data_val.csv'))
    expl_env = NormalizedBoxEnv(gym.make('MarketEnv-v0', raw_data=df_raw_data_train, returns=df_ret_train, features=df_feature_train,
                                         **expl_env_kwargs))

    eval_env = NormalizedBoxEnv(gym.make('MarketEnv-v0', raw_data=df_raw_data_val, returns=df_ret_val, features=df_feature_val,
                                         **eval_env_kwargs))

    def post_epoch_func(self, epoch):
        if not ENABLE_EPOCH_PLOT:
            return
        progress_csv = os.path.join(log_dir, 'progress.csv')
        df = pd.read_csv(progress_csv)
        final_kpis = FINAL_KPIS
        mean_kpis = MEAN_KPIS
        n = 50

        for kpi in mean_kpis:
            series = map(lambda s: df[f'{s}/env_infos/{kpi} Mean'], ['evaluation'])
            plot_ma(series=series, lables=['evaluation'], title=kpi, n=n)
            plt.savefig(os.path.join(log_dir, f'evaluation_{kpi}.png'))
            plt.close()

            series = map(lambda s: df[f'{s}/env_infos/{kpi} Mean'], ['exploration'])
            plot_ma(series=series, lables=['exploration'], title=kpi, n=n)
            plt.savefig(os.path.join(log_dir, f'exploration_{kpi}.png'))
            plt.close()



        for kpi in final_kpis:
            series = map(lambda s: df[f'{s}/env_infos/final/{kpi} Mean'], ['evaluation'])
            plot_ma(series=series, lables=['evaluation'], title=kpi, n=n)
            plt.savefig(os.path.join(log_dir, f'evaluation_{kpi}.png'))
            plt.close()

            series = map(lambda s: df[f'{s}/env_infos/final/{kpi} Mean'], ['exploration'])
            plot_ma(series=series, lables=['exploration'], title=kpi, n=n)
            plt.savefig(os.path.join(log_dir, f'exploration_{kpi}.png'))
            plt.close()


    trainer = get_trainer(env=eval_env, **trainer_kwargs)
    policy = trainer.policy
    eval_policy = MakeDeterministic(policy)
    #eval_policy = policy
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = MdpPathCollector(
        expl_env,
        policy,
    )
    replay_buffer = EnvReplayBuffer(
        variant['replay_buffer_size'],
        expl_env,
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        **variant['algorithm_kwargs']
    )
    algorithm.post_epoch_funcs = [post_epoch_func, ]
    algorithm.to(ptu.device)
    algorithm.train()


ptu.set_gpu_mode(True)
#ptu.set_gpu_mode(False)

fast_forward_scale = 1

variant = dict(
    version="normal",
    replay_buffer_size=int(1E6),
    trainer_kwargs=dict(
        algorithm="SAC", #Can be SAC, TD3, or DDPG
        hidden_sizes=[256, 256],
        reward_scale=1000,  # Only used by SAC
    ),
    expl_env_kwargs=dict(
        noise=True,
        state_scale=1,
        reward_func=sharpe_ratio_reward,
        reward_func_kwargs=dict(
            threshold=0.03,
            drop_only=False
        )
        ,
        trade_freq='months',
        trade_pecentage=0.2
    ),
    eval_env_kwargs=dict(
        noise=False,
        state_scale=1,
        reward_func=sharpe_ratio_reward,
        reward_func_kwargs=dict(
            threshold=0.03,
            drop_only=False
        ),
        trade_freq='months',
        trade_pecentage=1
    ),
    algorithm_kwargs=dict(
        num_epochs=EPOCH,
        num_eval_steps_per_epoch=int(1000/fast_forward_scale),
        num_trains_per_train_loop=int(3000/fast_forward_scale),
        num_expl_steps_per_train_loop=int(1000/fast_forward_scale),
        min_num_steps_before_training=int(1000/fast_forward_scale),
        max_path_length=int(1000/fast_forward_scale),
        batch_size=256,
    )
)

train_model(variant)
if not ENABLE_EPHCH_PLOT:
    from plot import post_epoch_func
    post_epoch_func(log_dir, 50)

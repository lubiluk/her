from spinup import ddpg_her_pytorch
import torch as th
import gym
import wrappers

env_fn = lambda : wrappers.DoneOnSuccessWrapper(gym.make('FetchPush-v1'), reward_offset=0)

ac_kwargs = dict(hidden_sizes=[64, 64, 64], activation=th.nn.ReLU)

logger_kwargs = dict(output_dir='data/fetch_push_ddpg_her_0', exp_name='fetch_push_ddpg_her_0')

ddpg_her_pytorch(
    env_fn=env_fn, 
    ac_kwargs=ac_kwargs, 
    steps_per_epoch=15000, 
    epochs=200, 
    batch_size=256,
    replay_size=1000000,
    start_steps=1000,
    gamma=0.95,
    q_lr=0.001,
    pi_lr=0.001,

    logger_kwargs=logger_kwargs)
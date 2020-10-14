from spinup import ddpg_her_pytorch
import torch as th
import gym

env_fn = lambda : gym.make('FetchPush-v1')

ac_kwargs = dict(hidden_sizes=[512, 512], activation=th.nn.ReLU)

logger_kwargs = dict(output_dir='data/push_0', exp_name='push_0')

ddpg_her_pytorch(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=1500, epochs=200, logger_kwargs=logger_kwargs)
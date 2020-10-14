from spinup import ddpg_pytorch
import torch as th
import gym

env_fn = lambda : gym.make('Pendulum-v0')

ac_kwargs = dict(hidden_sizes=[512, 512], activation=th.nn.ReLU)

logger_kwargs = dict(output_dir='data/pendulum_0', exp_name='pendulum_0')

ddpg_pytorch(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=1500, epochs=200, logger_kwargs=logger_kwargs)
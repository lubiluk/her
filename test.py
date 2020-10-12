from spinup import td3_pytorch
import torch as th
import gym

env_fn = lambda : gym.wrappers.FlattenObservation(gym.wrappers.FilterObservation(gym.make('FetchPush-v1', reward_type='dense'), ['observation', 'desired_goal']))

ac_kwargs = dict(hidden_sizes=[512, 512, 64], activation=th.nn.ReLU)

logger_kwargs = dict(output_dir='data', exp_name='test_0')

td3_pytorch(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, logger_kwargs=logger_kwargs)
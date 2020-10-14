from spinup.utils.test_policy import load_policy_and_env, run_policy

env, get_action = load_policy_and_env('data/pendulum_0', 'last', True)
run_policy(env, get_action, 0, 10, True)
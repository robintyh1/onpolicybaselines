import gym
import numpy as np
from gym import spaces

"""
wrapper for discretizing continuous action space
"""
def discretizing_wrapper(env, K):
    """
    # discretize each action dimension to K bins
    """
    unwrapped_env = env.unwrapped
    unwrapped_env.orig_step_ = unwrapped_env.step
    unwrapped_env.orig_reset_ = unwrapped_env.reset
    
    action_low, action_high = env.action_space.low, env.action_space.high
    naction = action_low.size
    action_table = np.reshape([np.linspace(action_low[i], action_high[i], K) for i in range(naction)], [naction, K])
    assert action_table.shape == (naction, K)

    def discretizing_reset():
        obs = unwrapped_env.orig_reset_()
        return obs

    def discretizing_step(action):
        # action is a sequence of discrete indices
        action_cont = action_table[np.arange(naction), action]
        obs, rew, done, info = unwrapped_env.orig_step_(action_cont)
        
        return (obs, rew, done, info)

    # change observation space
    env.action_space = spaces.MultiDiscrete([[0, K-1] for _ in range(naction)])

    unwrapped_env.step = discretizing_step
    unwrapped_env.reset = discretizing_reset

    return env

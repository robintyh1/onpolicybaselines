import numpy as np
import gym

def action_wrapper(env):
    """
    [0,1] -> [-1,1]
    """
    unwrapped_env = env.unwrapped
    unwrapped_env.orig_step = unwrapped_env.step

    def new_step(action):
        action = (action - 0.5) * 2.0
        obs, rew, done, info = unwrapped_env.orig_step(action)

        return (obs, rew, done, info)

    unwrapped_env.step = new_step

    return env
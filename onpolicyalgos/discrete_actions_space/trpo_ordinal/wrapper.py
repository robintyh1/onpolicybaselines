# define wrapper that converts discrete action space to continous action space
import gym
import numpy as np

# wrapper for openai gym - algorithmic envs
from gym.spaces import Discrete, Box

def _to_onehot(ob, n):
    # convert a categorical variable to one-hot vector
    # n is the max size
    z = np.zeros(n)
    z[ob] = 1.0
    return z


def algowrapper(env):
    # ob: Discrete -> Box (one hot)
    # ac: Tuple -> Discrete (flatten)
    unwrapped_env = env.unwrapped
    unwrapped_env.orig_step = unwrapped_env._step
    unwrapped_env.orig_reset = unwrapped_env._reset
    action_dim = [d.n for d in unwrapped_env.action_space.spaces]
    assert len(action_dim) == 3
    action_size = np.prod(action_dim)
    obs_size = unwrapped_env.observation_space.n
    
    def new_step(action):
        # action is a Discrete action
        assert action < action_size
        a1 = action // (action_dim[1] * action_dim[2])
        r1 = action % (action_dim[1] * action_dim[2])
        a2 = r1 // action_dim[2]
        a3 = r1 % action_dim[2]
        obs, rew, done, info = unwrapped_env.orig_step([a1,a2,a3])
        obs_onehot = _to_onehot(obs, obs_size)
        return obs_onehot, rew, done, info

    def new_reset():
        obs = unwrapped_env.orig_reset()
        obs_onehot = _to_onehot(obs, obs_size)
        return obs_onehot        

    unwrapped_env._step = new_step
    unwrapped_env._reset = new_reset

    env._orig_observation_space = unwrapped_env.observation_space
    env._orig_action_space = unwrapped_env.action_space
    env.observation_space = Box(np.zeros(obs_size), np.ones(obs_size))
    env.action_space = Discrete(action_size)
    return env


"""
wrapper for high dimensional environment
add additional dimensions into the observations (noise)
"""
def noise_wrapper(env, scale=0.1):
    """
    add noise to observations
    """
    unwrapped_env = env.unwrapped
    unwrapped_env.orig_step = unwrapped_env._step
    unwrapped_env.orig_reset = unwrapped_env._reset

    def noise_reset():
        obs = unwrapped_env.orig_reset()
        obs += np.random.randn(*env.observation_space.low.shape) * scale

        return obs

    def noise_step(action):
        obs, rew, done, info = unwrapped_env.orig_step(action)
        obs += np.random.randn(*env.observation_space.low.shape) * scale

        return (obs, rew, done, info)

    unwrapped_env._step = noise_step
    unwrapped_env._reset = noise_reset

    return env


"""
wrapper for time index
"""
from gym import spaces
def time_wrapper(env):
    """
    add noise to observations
    """
    unwrapped_env = env.unwrapped
    unwrapped_env.orig_step = unwrapped_env.step
    unwrapped_env.orig_reset = unwrapped_env.reset

    def time_reset():
        obs = unwrapped_env.orig_reset()
        obs = np.append(obs, 0.0)

        return obs

    def time_step(action):
        obs, rew, done, info = unwrapped_env.orig_step(action)
        obs = np.append(obs, env._elapsed_steps + 1.0)

        return (obs, rew, done, info)

    # change observation space
    env.observation_space = spaces.Box(-np.inf, np.inf, shape=(env.observation_space.high.size + 1,))

    unwrapped_env.step = time_step
    unwrapped_env.reset = time_reset

    return env


"""
wrapper for discretizing continuous action space
"""
def discretizing_wrapper(env, K):
    """
    # discretize each action dimension to K bins
    """
    unwrapped_env = env.unwrapped
    unwrapped_env.orig_step = unwrapped_env.step
    unwrapped_env.orig_reset = unwrapped_env.reset
    
    action_low, action_high = env.action_space.low, env.action_space.high
    naction = action_low.size
    action_table = np.reshape([np.linspace(action_low[i], action_high[i], K) for i in range(naction)], [naction, K])
    assert action_table.shape == (naction, K)

    def discretizing_reset():
        obs = unwrapped_env.orig_reset()
        return obs

    def discretizing_step(action):
        # action is a sequence of discrete indices
        action_cont = action_table[np.arange(naction), action]
        obs, rew, done, info = unwrapped_env.orig_step(action_cont)
        
        return (obs, rew, done, info)

    # change observation space
    env.action_space = spaces.MultiDiscrete([[0, K-1] for _ in range(naction)])

    unwrapped_env.step = discretizing_step
    unwrapped_env.reset = discretizing_reset

    return env
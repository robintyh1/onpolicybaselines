#!/usr/bin/env python
import argparse
from baselines import bench, logger

import os
import numpy as np
import pickle
import actionwrapper
import roboschool


def train(env_id, num_timesteps, seed, lr, entcoef, continue_train, nsteps):
    from baselines.common import set_global_seeds
    from baselines.common.vec_env.vec_normalize import VecNormalize
    import ppo2
    from policies import MlpPolicy
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    def make_env():
        env = gym.make(env_id)
        env = actionwrapper.action_wrapper(env)
        env = bench.Monitor(env, logger.get_dir())
        return env
    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy

    ppo2.learn(policy=policy, env=env, nsteps=nsteps, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=entcoef,
        lr=lr,
        cliprange=0.2,
        total_timesteps=num_timesteps)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--entcoef', type=float, default=0.0)
    parser.add_argument('--continue-train', type=int, default=0) # 1 for continued training
    parser.add_argument('--nsteps', type=int, default=2048)
    args = parser.parse_args()
    logger.configure()
    train(args.env, nsteps=args.nsteps, entcoef=args.entcoef, num_timesteps=args.num_timesteps, seed=args.seed, lr=args.lr, continue_train=args.continue_train)


if __name__ == '__main__':
    main()


#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
#import mujoco_py # Mujoco must come before other imports. https://openai.slack.com/archives/C1H6P3R7B/p1492828680631850
from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import gym
import logging
from baselines import logger
from policies import MlpPolicy
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
import trpo_mpi
import optimizers
import sys
import numpy as np
import os
import actionwrapper
import roboschool
        
def train(env_id, num_timesteps, seed, maxkl, cg_iters):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    env = actionwrapper.action_wrapper(env)
    #import noisewrapper
    #env = noisewrapper.robotics2mujoco_wrapper(env)
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=32, num_hid_layers=2)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=maxkl, cg_iters=cg_iters, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--maxkl', type=float, default=0.01)
    parser.add_argument('--iters', type=int, default=10)
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, maxkl=args.maxkl, cg_iters=args.iters)

if __name__ == '__main__':
    main()

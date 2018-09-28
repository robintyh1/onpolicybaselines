#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
import mujoco_py # Mujoco must come before other imports. https://openai.slack.com/archives/C1H6P3R7B/p1492828680631850
from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import gym
import logging
from baselines import logger
import trpo_implicit
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
from normalizingpolicies import TRPOImplicitPolicy
import sys
import numpy as np
import os


def train(env_id, num_timesteps, seed, num_units, num_layers, maxkl, continued, stepperbatch):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return TRPOImplicitPolicy(name=name, ob_space=env.observation_space, ac_space=env.action_space,
            hid_size=32, num_hid_layers=2, num_units=num_units, num_layers=num_layers)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    trpo_implicit.learn(env, policy_fn, timesteps_per_batch=stepperbatch, max_kl=maxkl, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3, callback=None)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--num-units', type=int, default=3)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--maxkl', type=float, default=0.01)
    parser.add_argument('--continued', type=int, default=0)
    parser.add_argument('--stepperbatch', type=int, default=1024)
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, stepperbatch=args.stepperbatch, seed=args.seed, num_units=args.num_units, num_layers=args.num_layers, maxkl=args.maxkl,
        continued=args.continued)


if __name__ == '__main__':
    main()

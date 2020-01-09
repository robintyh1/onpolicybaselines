# Code for AAAI 2020 paper: Discretizing Continuous Action Space for On-Policy Optimization

This code base contains all instructions and codes necessary to reproduce the results in the paper. We provide TRPO with Gaussian, Gaussian + tanh, Discrete, Ordinal and Beta policy. We also provide PPO with Gaussian, Gaussian + tanh, Discrete, Ordinal and Beta policy. Finally, we provide ACKTR with Gaussian and Discrete policy

## Dependencies
Need to install OpenAI baselines and tensorflow
```bash
pip install baselines
pip install --upgrade --ignore-installed https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.6.0-cp35-none-linux_x86_64.whl
```

## Examples
To run TRPO with discrete policy (with K=11 atomic actions per dimension)
```bash
python trpo_discrete/run_trpo_discrete.py --bins 11 --env Hopper-v1 --seed 100
```

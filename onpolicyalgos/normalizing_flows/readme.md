# Code for paper: Boosting Trust Region Policy Optimization with Normalizing Flows Policy

This code base contains all instructions and codes necessary to reproduce the results in the paper. We provide TRPO with Gaussian, GMM, Normalizing flows and Beta policy. We also provide ACKTR with Gaussian and Normalizing flows policy.


## Dependencies
Need to install OpenAI baselines and tensorflow
```bash
pip install baselines
pip install --upgrade --ignore-installed https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.6.0-cp35-none-linux_x86_64.whl
```

## Examples
To run TRPO with normalizing flows policy
```bash
python trpo_normflow/run_trpo_normflow.py --env Hopper-v1 --seed 100
```

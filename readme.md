# On-Policy Optimization Baselines for Deep Reinforcement Learning

On-Policy Optimization Baselines offer a suite of on-policy optimization algorithms, built on top of OpenAI [baselines](https://github.com/openai/baselines). In addition to the original on-policy optimization baselines, this repository offers implementations of trust region search algorithms (TRPO, ACKTR) combined with Gaussian Mixture Model (GMM) and Normalizing flows Policy.

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

## Citations
If you use this repo for academic research, you are highly encouraged to cite the following papers:

- Yunhao Tang, Shipra Agrawal. "[Boosting Trust Region Policy Optimization by Normalizing Flows Policy](https://arxiv.org/abs/1809.10326)". arXiv:1809.10326 [cs.AI], 2018.


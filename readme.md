# On-Policy Optimization Baselines for Deep Reinforcement Learning

On-Policy Optimization Baselines offer a suite of on-policy optimization algorithms, built on top of OpenAI [baselines](https://github.com/openai/baselines). In addition to the original on-policy optimization baselines, this repository offers implementations of trust region search algorithms (TRPO, ACKTR) combined with Gaussian Mixture Model (GMM) and Normalizing flows Policy. This repository also contains wrappers necessary for discretizing continuous action space for on-policy optimization.

These ideas are based on the following papers. Please find the code in the proper sub-directories.

Further, this repository provides some recent baselines (e.g. Beta distribution) as part of the comparison in the papers.

## Citations
If you use this repo for academic research, you are highly encouraged to cite the following papers:

- Yunhao Tang, Shipra Agrawal. "[Boosting Trust Region Policy Optimization by Normalizing Flows Policy](https://arxiv.org/abs/1809.10326)". arXiv:1809.10326 [cs.AI], 2018.
- Yunhao Tang, Shipra Agrawal. "Discretizing Continuous Action Space for On-Policy Optimization"(AAAI, 2020)

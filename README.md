# Pseudo-extended Markov Chain Monte Carlo

Thie repository contain the code related to the paper "Pseudo-extended Markov chain Monte Carlo" by Christopher Nemeth, Fredrik Lindsten, Maurizio Filippone and James Hensman

The repository contains files, each corresponding to the following examples from the paper:
 * two-dimensional mixture of 20 Gaussians 
 * Boltzmann machine relaxation
 * Complex targets, including the banana and flower distributions

For each of the folders there is a `.py` which runs the code. There also a `.stan` file which contains the STAN code for the model.

For the Boltzmann machine comparisons, please check out the supporting [code](https://github.com/matt-graham/continuously-tempered-hmc) for excellent [continuously tempered HMC](https://arxiv.org/abs/1704.03338) paper. The relaxation parameters were generated using the [code](https://github.com/matt-graham/boltzmann-machine-tools) produced by @matt-graham

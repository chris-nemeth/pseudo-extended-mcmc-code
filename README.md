# Pseudo-extended Markov Chain Monte Carlo

This repo contains the code related to the paper "Pseudo-extended Markov chain Monte Carlo" accepted to NeurIPS 2019.

This folder contains files, each corresponding to the following examples from the paper:
 * Mixture of univariate Gaussians
 * Mixture of 20 bivariate Gaussians 
 * Boltzmann machine relaxation
 * Sparse logistic regression with horseshoe priors

For each of the folders there is a `.py` which runs the code. There also a `.stan` file which contains the STAN code for the model.

For the Boltzmann machine comparisons, please check out the supporting [code](https://github.com/matt-graham/continuously-tempered-hmc) for the excellent [continuously tempered HMC](https://arxiv.org/abs/1704.03338) paper. The relaxation parameters were generated using the [code](https://github.com/matt-graham/boltzmann-machine-tools) produced by [@matt-graham](https://github.com/matt-graham)

Additionally, code for the RAM, EE and PT algorithms used in Section 4.1 follows from the Tak et al. (2018). A Repelling–Attracting Metropolis Algorithm for Multimodality. Journal of Computational and Graphical Statistics. 27(3), 479-490. Supporting code found [here](https://tandf.figshare.com/articles/A_Repelling-Attracting_Metropolis_Algorithm_for_Multimodality/5727080/1).

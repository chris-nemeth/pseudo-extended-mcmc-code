#This file runs the Boltmann machine relaxation example from the paper

import numpy as np
import pystan
import matplotlib.pyplot as plt
import random
import seaborn as sns
sns.set_style('white')

random.seed(42)
#---------------------------------------------------------------------------
#Functions

#Vectorised log-hyperbolic cosine
def log_cosh(y):
    return y + np.log(1+np.exp(-2*y)) - np.log(2)

#Log-target
def log_f(x,Q,b):
    return np.sum(log_cosh(np.matmul(Q,x.T)+b.reshape(-1,1)),0) - 0.5*np.diag(np.matmul(x,x.T))

#-----------------------------------------------------------------------------------
#Load the data

relaxation_list = np.load("relaxations.npz") #relaxation paramters
var_moms = np.load("var_moms.npz")           #variational approx. used for Graham and Storkey

Q = relaxation_list['Q']  # Relaxation Q matrix  parameters
b = relaxation_list['b']  # Relaxation  bias  vector  parameters
nb = b.size               # Number  of  dimension  in  Boltzmann  machine  binary  state
nr = Q.shape[1]           # Number  of  dimensions  in  relaxation  configuration  state
#---------------------------------------------------------------------------------------
#Standard HMC

dat = {'n_dim_b': nb,
       'n_dim_r': nr,
       'q': Q,
       'b': b}

iterations=10000

#Fit Stan model
fit = pystan.stan(file="boltzmann_target.stan", data=dat, iter=iterations, chains=1)

output = fit.extract()

sampsHMC = output['x'] #target samples

#-----------------------------------------------------------------------------
#Pseudo-extended MCMC
nParticles = 10

dat = {'n_dim_b': nb,
       'n_dim_r': nr,
       'q': Q,
       'b': b,
       'P': nParticles}

iterations=10000

#Fit Stan model
fit = pystan.stan(file="boltzmann_pseudo-extended.stan", data=dat, iter=iterations, chains=1)

output = fit.extract()

theta = output['x']
beta = output['beta']

#Calculate the weights of each particle 
weights = np.empty([iterations/2,nParticles])
for j in range(iterations/2):
    weights[j,:] = np.exp(log_f(theta[j],Q,b)*(1-beta[j]))

#Resample the particles \proto weights 
index = np.array([np.random.choice(a=nParticles, size=1, p=w/sum(w)) for w in weights])

pseudoSamples = np.array([theta[j,index[j].squeeze(),:]  for j in range(iterations/2)])

#-----------------------------------------------------------------
#A Stan implementation of the Graham and Storkey paper https://arxiv.org/abs/1704.03338

dat = {'n_dim_b': nb,
       'n_dim_r': nr,
       'q': Q,
       'b': b,
       'log_zeta':var_moms['var_log_norm'],
       'chol_sigma':var_moms['var_covar_chol'],
       'mu':var_moms['var_mean']}

iterations=10000

#Run Stan model
fit = pystan.stan(file="graham_storkey_ct.stan", data=dat, iter=iterations, chains=1)

output2 = fit.extract()

gs = output2['x']


#################################################################################
#------------------------------------------------------------------------------
################################################################################

#It's possible to compare the results against independent sampling using the Boltzmann tools library found here https://github.com/matt-graham/boltzmann-machine-tools

#independent_samples
import bmtools.exact.moments as mom
import bmtools.relaxations.gm_relaxations as gmr
import bmtools.utils as utils

rng = np.random.RandomState(201702)

#Model parameters
relax = np.load('params_and_moms.npz')

relaxation = gmr.IsotropicCovarianceGMRelaxation(relax['weights'], relax['biases'], True)
xs, _, _, _ = relaxation.independent_samples(10000, force=True, prng=rng)

#Plots
fig, axes = plt.subplots(1,4,sharex=True,sharey=True,figsize=(12, 8))
axes[0].plot(xs[:, -1], xs[:, -2], '.', ms=4, alpha=0.8)
axes[1].plot(sampsHMC[:, -1], sampsHMC[:, -2], '.', ms=4, alpha=0.8)
axes[2].plot(pseudoSamples[:, -1], pseudoSamples[:, -2], '.', ms=4, alpha=0.8)
axes[3].plot(gs[:, -1], gs[:, -2], '.', ms=4, alpha=0.8)
axes[0].set_title(r"Independent sampples")
axes[1].set_title(r"HMC sampler")
axes[2].set_title(r"Pseudo-extended")
axes[3].set_title(r"Graham and Storkey")
fig.tight_layout(pad=0)
plt.show()


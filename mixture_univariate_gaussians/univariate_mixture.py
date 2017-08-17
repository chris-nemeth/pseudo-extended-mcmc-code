#This files produces the MCMC samples for the simple univariate bimodal target example

import numpy as np
from matplotlib import pyplot as plt
import pystan
import seaborn as sns
sns.set_style('whitegrid')


###################################################################
#Target

mu = [-1.0,1.0]
sigma = np.sqrt([0.1,0.02])

def log_f(x):
    val = np.log(
        (0.5 / (2 * np.pi * sigma[0]**2)**0.5) * np.exp(-0.5 * ((x - mu[0]) / sigma[0])**2) +
        (0.5 / (2 * np.pi * sigma[1]**2)**0.5) * np.exp(-0.5 * ((x - mu[1]) / sigma[1])**2)
    )
    return val

#######################################################################
#Standard HMC

dat = {'mu': mu,
       'sigma': sigma}
 
iterations=10000

#Fit Stan model using standard HMC
fit = pystan.stan(file="bimodal_target.stan", data=dat, iter=iterations, chains=1)

output = fit.extract()

sampsHMC = output['theta']  #samples from the target

##############################################################
#Pseudo-extended MCMC
nParticles = 2

dat = {'P': nParticles,
       'mu': mu,
       'sigma': sigma}

iterations=10000

#Compile Stan model using pseudo-extended HMC
fit = pystan.stan(file="bimodal_pseudo-extended.stan", data=dat, iter=iterations, chains=1)

output = fit.extract()

theta = output['theta']
index = output['index']

#Resample the weighted samples
margSamples = np.array([theta[i,index[i].astype(int)]  for i in range(iterations/2)])

#############################################################################
#Plots


fig, axes = plt.subplots(1, 2, sharex=True, figsize=(6, 3))
xs = np.linspace(-3, 3, 200)
axes[0].plot(xs, np.exp(log_f(xs)))
axes[1].plot(xs, np.exp(log_f(xs)))
axes[0].hist(sampsHMC, 100, normed=True,alpha=0.8)
axes[1].hist(margSamples, 100, normed=True,alpha=0.8)
axes[0].set_title(r"Standard HMC")
axes[1].set_title(r"Pseudo-extended HMC")
plt.show()



#This script executes HMC (via STAN) on the Flower target

import numpy as np
import pystan
from matplotlib import pyplot as plt
import random
from numpy.linalg import norm
from scipy.stats import norm as normal
import seaborn as sns
sns.set_style('whitegrid')

random.seed(42)
#--------------------------------------------------------------------
#Target

r= 10
A = 6
w = 6
sigma = 1


def log_target(X,r,A,w,sigma):
    # compute all norms
    norms = norm(X) 
        
    # compute angles (second component first)
    angles = np.arctan2(X[1], X[0])
        
    # gaussian parameters
    mu = r + A * np.cos(w * angles)
        
    return normal.logpdf(norms,mu,sigma)


iterations = 10000

#Compile STAN models
hmcModel = pystan.StanModel(file="flower_target.stan")
pseudoModel = pystan.StanModel(file="pseudo-extended_flower.stan")


#-------------------------------------------------------------------
#HMC

data = {'r':r,
        'A':A,
        'w':w,
        'sigma':sigma}

fit1 = hmcModel.sampling(data=data,iter=iterations, chains=1)

output1 = fit1.extract()
hmcSamps = output1['x']

print(fit1)
fit1.plot()
plt.show()

#--------------------------------------------------------------------
#Pseudo-extended

data = {'N': 2,
        'r':r,
        'A':A,
        'w':w,
        'sigma':sigma}

fit2 = pseudoModel.sampling(data=data, iter=iterations, chains=1)

output2 = fit2.extract()
pseudoSamps = output2['z']

print(fit2)
fit2.plot('z')
plt.show()

#=============================================================
#Plots

fig, axes = plt.subplots(1, 2,sharex=True,sharey=True,figsize=(6,3))
xx = np.linspace(-20, 20, 300)
xx, yy = np.meshgrid(xx, xx)
X = np.vstack([xx.flatten(), yy.flatten()]).T
ll = [log_target(xi,r,A,w,sigma) for xi in X] 
axes[0].contour(xx, yy, np.exp(ll).reshape(*xx.shape), colors='b')
axes[0].scatter(hmcSamps[:, 0], hmcSamps[:, 1], color='k', alpha=0.3, lw=0)
axes[0].plot(hmcSamps[:, 0], hmcSamps[:, 1],alpha=0.3)
axes[0].set_title(r"HMC")
axes[1].contour(xx, yy, np.exp(ll).reshape(*xx.shape),colors='b')
axes[1].scatter(pseudoSamps[:, 0], pseudoSamps[:, 1], color='k', alpha=0.3, lw=0)
axes[1].plot(pseudoSamps[:, 0], pseudoSamps[:, 1],alpha=0.3)
axes[1].set_title(r"Pseudo-extended HMC")
plt.show()






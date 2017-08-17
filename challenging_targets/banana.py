#This script executes HMC (via STAN) on the Banana target
#
# X ~ N(0,Sigma), Sigma = diag(v,1...,1)
# X -> Y, where Y_2 = X_2 + b(X_1^2 - v), and Y_i = X_i,  
#
# pi(y,b,v) = N(y_1|0,v)N(y_2|b(y_1^2-v),1) \prod_{j=3}^{d}N(y_j|0,1)
#-------------------------------------------------------------------
#Libraries
import numpy as np
import pystan
import random
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import seaborn as sns
sns.set_style('whitegrid')

random.seed(42)

#--------------------------------------------------------------------
#Target

b = 0.1
v = 100.0


def log_target(X,b,v):
    Y = X
    Y[1] = X[1] - b * ((X[0] ** 2) - v)
    Y[0] = X[0] / np.sqrt(v)
    return multivariate_normal.logpdf(Y,np.zeros([2]),np.eye(2))


iterations = 10000

#Compile STAN models
hmcModel = pystan.StanModel(file="banana_target.stan")
pseudoModel = pystan.StanModel(file="pseudo-extended_banana.stan")


#-------------------------------------------------------------------
#HMC

data = {'b':b,
        'v':v}

fit1 = hmcModel.sampling(data=data,iter=iterations, chains=1)

output1 = fit1.extract()
hmcSamps = output1['x']

print(fit1)
fit1.plot()
plt.show()

#--------------------------------------------------------------------
#Pseudo-extended

data = {'N': 3,
        'b':b,
        'v':v}

fit2 = pseudoModel.sampling(data=data, iter=iterations, chains=1)

output2 = fit2.extract()
pseudoSamps = output2['z']

print(fit2)
fit2.plot('z')
plt.show()

#=============================================================
#Plots

fig, axes = plt.subplots(1, 2,sharex=True,sharey=True,figsize=(6,3))
xx = np.linspace(-50, 50, 300)
xx, yy = np.meshgrid(xx, xx)
X = np.vstack([xx.flatten(), yy.flatten()]).T
ll = [log_target(xi,b,v) for xi in X] 
axes[0].contour(xx, yy, np.exp(ll).reshape(*xx.shape), 100, colors='b')
axes[0].scatter(hmcSamps[:, 0], hmcSamps[:, 1], color='k', alpha=0.3, lw=0)
axes[0].plot(hmcSamps[:, 0], hmcSamps[:, 1],alpha=0.3)
axes[0].set_title(r"HMC")
axes[1].contour(xx, yy, np.exp(ll).reshape(*xx.shape), 100,colors='b')
axes[1].scatter(pseudoSamps[pseudoSamps[:,1]<50,0], pseudoSamps[pseudoSamps[:,1]<50,1], color='k', alpha=0.3, lw=0)
axes[1].plot(pseudoSamps[pseudoSamps[:,1]<50,0], pseudoSamps[pseudoSamps[:,1]<50,1],alpha=0.3)
axes[1].set_title(r"Pseudo-extended HMC")
plt.xlim([-30,30])
plt.ylim([-15,50])
plt.show()



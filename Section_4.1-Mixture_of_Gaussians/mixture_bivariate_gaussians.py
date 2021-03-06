#This file compares standard HMC against pseudo-extended HMC on a 2D mixture of 20 Gaussians example. The example is taken from "A Repelling-Attracting Metropolis Algorithm for Multimodality," Tak et al. (2016) and Kou, S. C., Zhou, Q., and Wong, W. H. (2006). Discussion paper: Equi-energy sampler with applications in statistical inference and statistical mechanics. The Annals of Statistics, 34(4):1581–1619.

import numpy as np
from matplotlib import pyplot as plt
import pystan
import random
import seaborn as sns
sns.set_style('whitegrid')

random.seed(42)

#-----------------------------------------------------------------------
#Scenario A

mu = np.array([[2.18,5.76],[8.67,9.59],[4.24,8.48],[8.41,1.68],[3.93,8.82],[3.25,3.47],[1.70,0.50],[4.59,5.60],[6.91,5.81],[6.87,5.40],[5.41,2.65],[2.70,7.88],[4.98,3.70],[1.14,2.39],[8.33,9.50],[4.93,1.5],[1.83,0.09],[2.26,0.31],[5.54,6.86],[1.69,8.11]])

num_mixtures = 20

#Log target f(x)
def log_f(x):
    w = (1.0/num_mixtures)*np.ones(num_mixtures)
    v = 1.0/(10.0)*np.ones(num_mixtures)
    return np.log(np.sum((w/v**2)*np.exp(-1.0/(2.0*v**2)*np.sum(np.square(x-mu),1))))


dim =2              # mixture of bivariate Gaussians
num_mixtures = 20   # number of mixture components
iterations=10000    # number of MCMC iterations


#Stan models pre-compiled for case A
hmcModel = pystan.StanModel(file="multimodal_target_1.stan")
pseudoModel = pystan.StanModel(file="pseudo-extended_1.stan")

#------------------------------------------------------------------
#Standard HMC

dat = {'K': 20,
       'd': 2,
       'mu':mu,
       'sigma':1.0/(100.0)*np.ones(num_mixtures)}

#Fit Stan model
fit = hmcModel.sampling(data=dat, iter=iterations, chains=1)

output = fit.extract()

sampsHMC = output['theta']

#------------------------------------------------------------------------
#Pseudo-extended MCMC with tempered proposal

nParticles = 2

dat = {'K': 20,
       'd': 2,
       'N': nParticles,
       'mu':mu,
       'sigma':1.0/(100.0)*np.ones(num_mixtures)}


#Fit Stan model
fit = pseudoModel.sampling(data=dat, iter=iterations, chains=1)
 #The NUTS tuner can have issues when beta is close to zero as the target is not well-define. Modifying the STAN tuning parameters can help (i.e. control=dict(adapt_delta=0.9999,max_treedepth=20) )

output = fit.extract()

theta = output['theta']
beta = output['beta']

weights = np.empty([iterations//2,nParticles])
for i in range(iterations//2):
    weights[i,:] = [np.exp(log_f(x)*(1-b)) for (x,b) in zip(theta[i],beta[i])]
index = np.array([np.random.choice(a=nParticles, size=1, p=w/sum(w)) for w in weights])
pseudoHMC = np.array([theta[i,index[i].squeeze(),:]  for i in range(iterations//2)])

#-----------------------------------------------------------------------------------
#====================================================================================
#-----------------------------------------------------------------------------------
#Scenario B

mu = np.array([[2.18,5.76],[8.67,9.59],[4.24,8.48],[8.41,1.68],[3.93,8.82],[3.25,3.47],[1.70,0.50],[4.59,5.60],[6.91,5.81],[6.87,5.40],[5.41,2.65],[2.70,7.88],[4.98,3.70],[1.14,2.39],[8.33,9.50],[4.93,1.5],[1.83,0.09],[2.26,0.31],[5.54,6.86],[1.69,8.11]])

num_mixtures = 20

#Log target f(x)
def log_f2(x):
    w = 1/np.sqrt(np.sum(np.square((mu-[5,5])),1))
    v = np.sqrt(np.sum(np.square((mu-[5,5])),1))/20
    return np.log(np.sum((w/v)*np.exp(-1.0/(2.0*v)*np.sum(np.square(x-mu),1))))


dim = 2              # mixutre of bivariate Gaussians
num_mixtures = 20    # number of mixture components
iterations = 10000   # number of MCMC iterations


#Stan models pre-compiled for case B
hmcModel2 = pystan.StanModel(file="multimodal_target_2.stan")
pseudoModel2 = pystan.StanModel(file="pseudo-extended_2.stan")

#------------------------------------------------------------------
#Standard HMC

dat = {'K': 20,
       'd': 2,
       'w': 1/np.sqrt(np.sum(np.square((mu-[5,5])),1)),
       'mu':mu,
       'sigma': np.sqrt(np.sum(np.square((mu-[5,5])),1))/20}


fit = hmcModel2.sampling(data=dat, iter=iterations, chains=1)
output = fit.extract()
sampsHMC2 = output['theta']  #target samples

#------------------------------------------------------------------------
#Pseudo-extended MCMC with tempered proposal

nParticles = 2

dat = {'K': 20,
       'd': 2,
       'N': nParticles,
       'w': 1/np.sqrt(np.sum(np.square((mu-[5,5])),1)),
       'mu':mu,
       'sigma': np.sqrt(np.sum(np.square((mu-[5,5])),1))/20}

#Run Stan model
fit = pseudoModel2.sampling(data=dat, iter=iterations, chains=1)
output = fit.extract()
theta = output['theta']
betas = output['beta']

#Weight and resample 
weights = np.empty([iterations//2,nParticles])
for i in range(iterations//2):
    weights[i,:] = [np.exp(log_f2(x)*(1-b)) for (x,b) in zip(theta[i],betas[i])]
index = np.array([np.random.choice(a=nParticles, size=1, p=w/sum(w)) for w in weights])
pseudoHMC2 = np.array([theta[i,index[i].squeeze(),:]  for i in range(iterations//2)])

#-----------------------------------------------------------------------------------
#====================================================================================
#-----------------------------------------------------------------------------------

#Plot the results

fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True)
xx = np.linspace(-1, 10, 100)
xx, yy = np.meshgrid(xx, xx)
Xtest = np.vstack([xx.flatten(), yy.flatten()]).T
ll = [log_f(xi) for xi in Xtest]
ll2 = [log_f2(xi) for xi in Xtest] 
axes[0,0].contour(xx, yy, np.exp(ll).reshape(*xx.shape), colors='b')
axes[0,1].contour(xx, yy, np.exp(ll).reshape(*xx.shape), colors='b')
axes[1,0].contour(xx, yy, np.exp(ll2).reshape(*xx.shape), 80, colors='b')
axes[1,1].contour(xx, yy, np.exp(ll2).reshape(*xx.shape), 80, colors='b')

axes[0,0].scatter(sampsHMC[:, 0], sampsHMC[:, 1], color='k', alpha=0.2, lw=0)
axes[0,0].set_title(r"HMC sampler")
axes[0,0].set_xlabel(r"$x_1$")
axes[0,0].set_ylabel(r"$x_2$")
axes[0,1].scatter(pseudoHMC[:, 0], pseudoHMC[:, 1], color='k', alpha=0.2, lw=0)
axes[0,1].set_title(r"Pseudo-extended HMC sampler")
axes[0,1].set_xlabel(r"$x_1$")
axes[0,1].set_ylabel(r"$x_2$")
axes[1,0].scatter(sampsHMC2[:, 0], sampsHMC2[:, 1], color='k', alpha=0.2, lw=0)
axes[1,0].set_title(r"HMC sampler")
axes[1,0].set_xlabel(r"$x_1$")
axes[1,0].set_ylabel(r"$x_2$")
axes[1,1].scatter(pseudoHMC2[:, 0], pseudoHMC2[:, 1], color='k', alpha=0.2, lw=0)
axes[1,1].set_title(r"Pseudo-extended HMC sampler")
axes[1,1].set_xlabel(r"$x_1$")
axes[1,1].set_ylabel(r"$x_2$")
fig.tight_layout(pad=1)
plt.show()

#---------------------------------------------------------------------------
#=========================================================================
#------------------------------------------------------------------------
#Additional simulations (results of which appear in the Supplementary Material) for the case where \beta is fixed. This code performs a full simulation over the range of beta values with varying number of particles

#Scenario A

mu = np.array([[2.18,5.76],[8.67,9.59],[4.24,8.48],[8.41,1.68],[3.93,8.82],[3.25,3.47],[1.70,0.50],[4.59,5.60],[6.91,5.81],[6.87,5.40],[5.41,2.65],[2.70,7.88],[4.98,3.70],[1.14,2.39],[8.33,9.50],[4.93,1.5],[1.83,0.09],[2.26,0.31],[5.54,6.86],[1.69,8.11]])


num_mixtures = 20

#Log target f(x)
def log_f(x):
    w = (1.0/num_mixtures)*np.ones(num_mixtures)
    v = 1.0/(10.0)*np.ones(num_mixtures)
    return np.log(np.sum((w/v**2)*np.exp(-1.0/(2.0*v**2)*np.sum(np.square(x-mu),1))))

dim = 2              # mixutre of bivariate Gaussians
num_mixtures = 20    # number of mixture components
iterations = 100000   # number of MCMC iterations


#Stan model
pseudoModel = pystan.StanModel(file="pseudo-extended-fixedBeta.stan")

#------------------------------------------------------------------------
#Pseudo-extended MCMC with tempered proposal and fixed beta

nParticles = [2,5,10,20]
betas = np.arange(0.1,1.0,0.1)

for t in range(len(betas)):
    for p in range(4):
        print('Particles:',nParticles[p],'Beta:',betas[t])
        id_string = 'pseudo_' + str(nParticles[p])+'_particles_fixedBeta_' + str(betas[t]) 
        dat = {'K': 20,
               'd': 2,
               'N': nParticles[p],
               'mu':mu,
               'sigma':1.0/(100.0)*np.ones(num_mixtures),
               'beta': betas[t]}
        #Fit Stan model
        tic=timeit.default_timer()
        fit = pseudoModel.sampling(data=dat, iter=iterations, chains=1)
        output = fit.extract()
        theta = output['theta']
        weights = np.empty([iterations//2,nParticles[p]])
        for i in range(iterations//2):
            weights[i,:] = [np.exp(log_f(x)*(1-betas[t])) for x in theta[i]]
        index = np.array([np.random.choice(a=nParticles[p], size=1, p=w/sum(w)) for w in weights])
        pseudoHMC = np.array([theta[i,index[i].squeeze(),:]  for i in range(iterations//2)])
        toc=timeit.default_timer()
        np.savetxt(path + id_string+'.txt',[toc-tic])
        with open(path + id_string + '.pickle', 'wb') as f:
            pickle.dump(pseudoHMC,f)


#-------------------------------------------------------------------
#====================================================================

#Scenario B

#Stan model

pseudoModel = pystan.StanModel(file="pseudo-extended-fixedBeta2.stan")

#------------------------------------------------------------------------
#Pseudo-extended MCMC with tempered proposal and fixed beta

nParticles = [2,5,10,20]
betas = np.arange(0.1,1.0,0.1)

for t in range(len(betas)):
    for p in range(np.size(nParticles)):
        print('Particles:',nParticles[p],'Beta:',betas[t])
        id_string = 'pseudo2_' + str(nParticles[p])+'_particles_fixedBeta_' + str(betas[t]) 
        dat = {'K': 20,
               'd': 2,
               'N': nParticles[p],
               'w': 1/np.sqrt(np.sum(np.square((mu-[5,5])),1)),
               'mu':mu,
               'sigma': np.sqrt(np.sum(np.square((mu-[5,5])),1))/20,
               'beta': betas[t]}
        #Fit Stan model
        tic=timeit.default_timer()
        fit = pseudoModel.sampling(data=dat, iter=iterations, chains=1)
        output = fit.extract()
        theta = output['theta']
        weights = np.empty([iterations//2,nParticles[p]])
        for i in range(iterations//2):
            weights[i,:] = [np.exp(log_f(x)*(1-betas[t])) for x in theta[i]]
        index = np.array([np.random.choice(a=nParticles[p], size=1, p=w/sum(w)) for w in weights])
        pseudoHMC = np.array([theta[i,index[i].squeeze(),:]  for i in range(iterations//2)])
        toc=timeit.default_timer()
        np.savetxt(path + id_string+'.txt',[toc-tic])
        with open(path + id_string + '.pickle', 'wb') as f:
            pickle.dump(pseudoHMC,f)


#-------------------------------------------------------------------


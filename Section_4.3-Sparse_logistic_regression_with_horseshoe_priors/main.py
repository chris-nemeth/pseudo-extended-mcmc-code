#This script tests out the horseshoe prior for variable selection based on Piironen and Vehtari (2017).

import scipy.io as spio
import numpy as np
import pystan 
from scipy.stats import cauchy, norm 
from matplotlib import pyplot as plt
import csv

#Load the data
mat = spio.loadmat('colon.mat', squeeze_me=True) #or 'prostate.mat' or 'leukemia.mat'

Y = mat['Y']  #responses
Y = (Y+1)/2
X = mat['X']  #covariates
X = X/2

num_obs, dim = X.shape

#Split into test and training 80/20
train = set(np.random.choice(num_obs,np.floor(0.8*num_obs).astype(int),replace=False))
idd = set(range(num_obs))
test = list(idd - train)
train = list(train)

Xtrain = X[train]
Xtest  = X[test]
Ytrain = Y[train]
Ytest  = Y[test]

num_obs, dim = Xtrain.shape

p0 = 3.0 #colon
#p0 = 200.0 #prostate
#p0 = 55.0 #leukemia

#Functions
def inv_logit(f):
    return 1.0/(1.0+np.exp(-f))

def log_like(Y,x,beta0,beta):
    f = inv_logit(beta0+np.matmul(x,beta))
    logl = np.nansum(Y*np.log(f) + (1.0-Y)*np.log(1.0-f))
    return logl

def log_priors(beta0,beta,tau,lambdas,tau0,sigma,scale_icept):
    return np.sum(cauchy.logpdf(tau,0,tau0) + cauchy.logpdf(lambdas,0,sigma) + norm.logpdf(beta0,0,scale_icept) + norm.logpdf(beta,tau*lambdas))

def log_post(Y,x,beta0,beta,tau,lambdas,tau0,sigma,scale_icept):
    return log_like(Y,x,beta0,beta) + log_priors(beta0,beta,tau,lambdas,tau0,sigma,scale_icept)

def log_pred(Y,x,beta0,beta):
    f = inv_logit(beta0+np.matmul(x,beta))
    logp = Y*np.log(f) + (1.0-Y)*np.log(1.0-f)
    return np.nansum(logp)


#Compile the Stan code
hmcModel = pystan.StanModel(file="hmc_regularised_horseshoe_classification.stan")
peModel = pystan.StanModel(file="pseudo-extended_regularised_horseshoe_classification.stan")
peModelFixedBeta = pystan.StanModel(file="pseudo-extended_regularised_horseshoe_classification_fixedBeta.stan") 

#-----------------------------------------------------------------------
#Run standard Stan model

iterations = 1000

#regularised
sigma = 2
data = {'n': num_obs,
        'd': dim,
        'y': Ytrain.astype(int),
        'x': Xtrain,
        'scale_icept': 10.0,
        'scale_global': (p0/((dim-p0))*(sigma/np.sqrt(num_obs))),
        'nu_global': 1.0,
        'nu_local': 1.0,
        'slab_scale': 2.0,
        'slab_df': 3}

fit1 = hmcModel.sampling(data=data, iter=iterations, chains=1)
output1 = fit1.extract()
beta0HMC = output1['beta0']
betaHMC = output1['beta']

#---------------------------------------------------------------------
#Run standard pseudo-extended model

num_particles = 2

#regularised
data = {'n': num_obs,
        'd': dim,
        'y': Ytrain.astype(int),
        'x': Xtrain,
        'scale_icept': 10.0,
        'scale_global': (p0/((dim-p0))*(sigma/np.sqrt(num_obs))),
        'nu_global': 1.0,
        'nu_local': 1.0,
        'slab_scale': 2.0,
        'slab_df': 3,
        'N': num_particles}


fit2 = peModel.sampling(data=data, iter=iterations, chains=1)
output2 = fit2.extract()
beta0PE = output2['beta0']
betaPE = output2['beta']
idx = output2['index']
beta0PE = np.array([beta0PE[i,idx[i].astype(int)]  for i in range(iterations//2)])
betaPE = np.array([betaPE[i,idx[i].astype(int),:]  for i in range(iterations//2)])

#------------------------------------------------------------------
#Run pseudo-extended model with fixed beta

num_particles = 2
gamma = 0.25 #what is referred to in the paper as \beta

#regularised
data = {'n': num_obs,
        'd': dim,
        'y': Ytrain.astype(int),
        'x': Xtrain,
        'scale_icept': 10.0,
        'scale_global': (p0/((dim-p0))*(sigma/np.sqrt(num_obs))),
        'nu_global': 1.0,
        'nu_local': 1.0,
        'slab_scale': 2.0,
        'slab_df': 3,
        'N': num_particles,
        'gamma': gamma}}

fit3 = peModelFixedBeta.sampling(data=data, iter=iterations, chains=1)
output3 = fit3.extract()
beta0PEfixedBeta = output3['beta0']
betaPEfixedBeta = output3['beta']
idx = output3['index']
beta0PEfixedBeta = np.array([beta0PEfixedBeta[i,idx[i].astype(int)]  for i in range(iterations//2)])
betaPEfixedBeta = np.array([betaPEfixedBeta[i,idx[i].astype(int),:]  for i in range(iterations//2)])

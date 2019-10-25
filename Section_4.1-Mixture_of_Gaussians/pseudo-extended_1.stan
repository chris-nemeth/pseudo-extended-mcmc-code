//This file is a pseudo-marginal implementation of the multimodal model with tempered proposal under Scenario (a)

data {	           // read in the data
  int<lower=1> K;         // number of mixture components
  int<lower=1> d;         // dimension
  int N;	          // number of particles
  row_vector[d] mu[K];         // location parameters
  vector[K] sigma;         // scale parameters
}

parameters {	           // model parameters
  row_vector[d] theta[N];
  vector<lower=0.001,upper=1>[N] beta;  //lower bounded to prevent beta=0
}	


model {	                     // model for the posterior
  vector[N] num;
  vector[N] denom;
  vector[N] ratio;  
  real min_lr;

  real ps[K];

  for (i in 1:N) {
        theta[i] ~ uniform(-10.0,10.0); //prior for theta
    }

  for(i in 1:N){
    for(k in 1:K){
      ps[k] = log(1.0/K) + multi_normal_lpdf(theta[i]|mu[k],diag_matrix(rep_vector(sigma[k],d)));
    }
    num[i] = log_sum_exp(ps);
    denom[i] = beta[i]*num[i]; 
  }
  ratio = num - denom;
  min_lr = min(ratio);
  ratio = ratio - min_lr;
  target += log_sum_exp(ratio) + min_lr - log(N) + sum(denom);
}	


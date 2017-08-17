//This file is a pseudo-extended Stan implementation of the bimodal model, with tempered proposal

data {	           // read in the data
  int P;           // number of particles
  vector[2] mu;    // mixture locations
  vector[2] sigma; // mixture standard deviations
}	

parameters {	           // model parameters
  vector[P] theta;
  vector<lower=0,upper=1>[P] beta; //temper
}	


model {	                     // model for the posterior
  vector[P] num;
  vector[P] denom;
  vector[P] ratio;
  real min_lr;

  real ps[2];

  for(i in 1:P){
    for(k in 1:2){
      ps[k] = log(0.5) + normal_lpdf(theta[i]|mu[k],sigma[k]);
    }
    num[i] = log_sum_exp(ps);
    denom[i] = beta[i]*num[i];
  }
  ratio = num - denom;
  min_lr = min(ratio);
  ratio = ratio - min_lr;
  target += log_sum_exp(ratio) + min_lr - log(P) + sum(denom);
}	

#Post-hoc reweighting
generated quantities{
  int<lower=0,upper=P-1> index;
  vector[P] weights;

  vector[P] num;
  vector[P] denom;
  vector[P] ratio;

  real ps[2];

  for(i in 1:P){
    for(k in 1:2){
      ps[k] = log(0.5) + normal_lpdf(theta[i]|mu[k],sigma[k]);
    }
    num[i] = log_sum_exp(ps);
    denom[i] = beta[i]*num[i];
  }
  
  ratio = num - denom;
  weights = exp(ratio - max(ratio));
  weights = weights/sum(weights);
  index = categorical_rng(weights)-1;
}

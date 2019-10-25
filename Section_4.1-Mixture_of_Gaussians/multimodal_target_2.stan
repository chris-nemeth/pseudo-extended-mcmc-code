//HMC implementation of the multimodal model under Scenario (b)

data {	           // read in the data
  int<lower=1> K;         // number of mixture components
  int<lower=1> d;         // dimension
  vector[K] w;           // weights
  row_vector[d] mu[K];         // location parameters
  vector[K] sigma;         // scale parameters
}

parameters {	           // model parameters
  row_vector[d] theta;
}	


model {	                     // model for the posterior

  real ps[K];

   for(k in 1:K){
      ps[k] = log(w[k]) + multi_normal_lpdf(theta|mu[k],diag_matrix(rep_vector(sigma[k],d)));
    }
    target += log_sum_exp(ps);
}	


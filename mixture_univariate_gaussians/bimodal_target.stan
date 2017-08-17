//This file is a Stan implementation of the bimodal model

data {	           // read in the data
  vector[2] mu;    // mixture locations
  vector[2] sigma; // mixture standard deviations
}	

parameters {	           // model parameters
  real theta;
}	


model {	                     // model for the posterior

  real ps[2];

    for(k in 1:2){
      ps[k] = log(0.5) + normal_lpdf(theta|mu[k],sigma[k]);
    }

  target += log_sum_exp(ps);
}	


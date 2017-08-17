//Stan implementation of the banana target using the pseudo-extended sampler

data{
 int<lower=1> N;   // number of pseudo-samples
 real b; 
 real v; 
}

parameters{
  row_vector[2] x[N];
  vector<lower=0,upper=1>[N] beta;
}

transformed parameters{
  row_vector[2] y[N];
  
  y = x;
  for(i in 1:N){  
    y[i][2] = x[i][2] - b*(x[i][1]^2-v);
    y[i][1] = x[i][1]/sqrt(v);
  }
}

model{
  vector[N] num;
  vector[N] denom;
  vector[N] ratio;  
  real min_lr;

  for(i in 1:N){
     num[i] = multi_normal_lpdf(y[i]|rep_vector(0,2),diag_matrix(rep_vector(1.0,2)));
     denom[i] = beta[i]*num[i];
  }
  ratio = num - denom;
  min_lr = min(ratio);
  ratio = ratio - min_lr;
  target += log_sum_exp(ratio) + min_lr - log(N) + sum(denom);
}

generated quantities{
  int<lower=0,upper=N-1> index;
  vector[N] weights;

  row_vector[2] z;

  vector[N] num;
  vector[N] denom;
  vector[N] ratio;
  real min_lr;

  for(i in 1:N){
     num[i] = multi_normal_lpdf(y[i]|rep_vector(0,2),diag_matrix(rep_vector(1.0,2)));
     denom[i] = beta[i]*num[i];
  }
  
  ratio = num - denom;
  weights = exp(ratio - max(ratio));
  weights = weights/sum(weights);
  index = categorical_rng(weights)-1;
  z = x[index+1];
}


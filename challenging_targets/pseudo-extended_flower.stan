//Stan implementation of the flower target using the pseudo-extended sampler

data{
 int<lower=1> N;   // number of pseudo-samples
 real r;           // radius
 real A;           // amplitude
 real w;           // frequency
 real sigma;       // standard deviation
}

parameters{
  row_vector[2] x[N];
  vector<lower=0,upper=1>[N] beta;
}

model{
  vector[N] num;
  vector[N] denom;
  vector[N] ratio;  
  real min_lr;

  vector[N] norm;
  vector[N] angle;
  vector[N] mu;

  for(i in 1:N){
  	norm[i] = sqrt(x[i][1]^2+x[i][2]^2);
	angle[i] = atan2(x[i][2],x[i][1]);
	mu[i] = r + A*cos(w*angle[i]);

	num[i] = normal_lpdf(norm[i]|mu[i],sigma);
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

  vector[N] norm;
  vector[N] angle;
  vector[N] mu;

  for(i in 1:N){
  	norm[i] = sqrt(x[i][1]^2+x[i][2]^2);
	angle[i] = atan2(x[i][2],x[i][1]);
	mu[i] = r + A*cos(w*angle[i]);

	num[i] = normal_lpdf(norm[i]|mu[i],sigma);
	denom[i] = beta[i]*num[i];
  }
  ratio = num - denom;
  weights = exp(ratio - max(ratio));
  weights = weights/sum(weights);
  index = categorical_rng(weights)-1;
  z = x[index+1];
}


//Stan code from Piironen and Vehtari (2017) regularised horseshoe with fixed beta (gamma).

data {
  int < lower =0 > n ; // number of observations
  int < lower =0 > d ; // number of predictors
  int < lower =0 , upper =1 > y[ n ]; // outputs
  matrix [n ,d] x; // inputs
  real < lower =0 > scale_icept ; // prior std for the intercept
  real < lower =0 > scale_global ; // scale for the half - t prior for tau
  real < lower =1 > nu_global ; // degrees of freedom for the half - t priors for tau
  real < lower =1 > nu_local ; // degrees of freedom for the half - t priors for lambdas
  // ( nu_local = 1 corresponds to the horseshoe )
  real < lower =0 > slab_scale; // slab scale for the regularized horseshoe
  real < lower =0 > slab_df;  //slab degrees of freedom for the regularized horseshoe
  int<lower=1> N;   //number of pseudo-samples
  real gamma;      //temper parameter
}

parameters {
  vector[N] beta0 ;
  vector[d] z[N] ;
  real<lower=0> tau;  //global shrinkage
  vector<lower=0>[d] lambda; //local shrinkage
  real<lower=0> caux ;
}

transformed parameters {
  vector<lower=0>[d] lambda_tilde; // ’truncated’ local shrinkage parameter
  real<lower=0> c; // slab scale
  vector[d] beta[N]; // regression coefficients
  matrix[n,N] f; // latent function values

    c = slab_scale * sqrt(caux);
    lambda_tilde = sqrt(c^2*square(lambda) ./ (c^2 + tau^2*square(lambda)));

  for(i in 1:N){
    beta[i] = z[i] .* lambda_tilde * tau ;
    f[,i] = beta0[i] + x * beta[i] ;
  }
}

model {
  vector[N] num;
  vector[N] denom;
  vector[N] ratio;  
  real min_lr;

  lambda ~ student_t ( nu_local , 0 , 1);
  tau ~ student_t ( nu_global , 0 , scale_global);
  caux ~ inv_gamma (0.5* slab_df , 0.5* slab_df );

  for(i in 1:N){
  //  half-t priors for lambdas and tau, and inverse-gamma for c^2
  z[i] ~ normal(0,1);
  beta0[i] ~ normal (0 , scale_icept );

  num[i] = bernoulli_logit_lpmf(y|f[,i]);
  denom[i] = gamma*num[i];
  }

  ratio = num - denom;
  min_lr = min(ratio);
  ratio = ratio - min_lr;
  target += log_sum_exp(ratio) + min_lr - log(N) + sum(denom);
}

generated quantities {
  int<lower=0,upper=N-1> index;
  vector[N] weights;

  vector[N] num;
  vector[N] denom;
  vector[N] ratio;  

  for(i in 1:N){
    num[i] = bernoulli_logit_lpmf(y|f[,i]);
    denom[i] = gamma*num[i];
  }

  ratio = num - denom;
  weights = exp(ratio - max(ratio));
  weights = weights/sum(weights);
  index = categorical_rng(weights)-1;
}

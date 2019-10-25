//Stan code from Piironen and Vehtari (2017) regularised horseshoe.

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
}

parameters {
  real beta0;
  vector[d] z;
  real<lower=0> tau;  //global shrinkage
  vector<lower=0>[d] lambda; //local shrinkage
  real<lower=0> caux;
}

transformed parameters {
  vector [ d ] beta ; // regression coefficients
  vector < lower =0 >[ d ] lambda_tilde ; // ’truncated’ local shrinkage parameter
  real < lower =0 > c ; // slab scale
  vector [ n ] f ; // latent function values
  c = slab_scale * sqrt ( caux );
  lambda_tilde = sqrt ( c ^2 * square ( lambda ) ./ ( c ^2 + tau ^2* square ( lambda )) );
  beta = z .* lambda_tilde * tau ;
  f = beta0 + x * beta ;
}

model {
  //  half-t priors for lambdas and tau, and inverse-gamma for c^2
  z ~ normal (0 , 1);
  lambda ~ student_t ( nu_local , 0 , 1);
  tau ~ student_t ( nu_global , 0 , scale_global);
  caux ~ inv_gamma (0.5* slab_df , 0.5* slab_df );
  beta0 ~ normal (0 , scale_icept );
  target += bernoulli_logit_lpmf(y|f);   // observation model
}

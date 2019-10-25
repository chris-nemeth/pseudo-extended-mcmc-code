//This file is an implementation of Graham and Storkey's continuous tempering for the boltzmann relaxation

functions{
  // Vectorised log hyperbolic cosine helper.
  vector log_cosh(vector y){
    return y + log(1 + exp(-2*y))-log(2);
  }

  // Log probability density of Boltzmann machine relaxation.
  real bm_relaxation_lpdf(vector x, matrix q, vector b){
    return sum(log_cosh(q*x + b))-0.5*x'*x;
  }
}

data{
  // Number of dimension in Boltzmann machine binary state.
  int<lower=0> n_dim_b;
  // Number of dimensions in relaxation configuration state.
  int<lower=0> n_dim_r;
  // Relaxation Q matrix parameters.
  matrix[n_dim_b,n_dim_r] q;
  // Relaxation bias vector parameters.
  vector[n_dim_b] b;
  // Target log normalisation constant approximation.
  real log_zeta;
  // Covariance matrix of Gaussian approximation to target.
  cholesky_factor_cov[n_dim_r, n_dim_r] chol_sigma;
  // Mean vector of Gaussian approximation to target.
  vector[n_dim_r] mu;
}


parameters{
  // Configuration state.
  vector[n_dim_r] x;
  real<lower=0, upper=1> beta;
}


model {
  // Inverse temperature weighted target density term.
  target += beta*bm_relaxation_lpdf(x|q,b)-beta*log_zeta;
  // Inverse temperature weighted base density term.
  target += (1-beta)*multi_normal_cholesky_lpdf(x|mu,chol_sigma);
}

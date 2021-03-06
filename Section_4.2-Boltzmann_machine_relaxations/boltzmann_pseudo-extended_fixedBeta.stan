//Pseudo-extended HMC implementation on Boltzmann machine with fixed beta

functions{
//  Vectorised  log  hyperbolic  cosine  helper .
vector log_cosh(vector y){
       return y + log(1 + exp(-2*y))-log(2);
}

//  Log  probability  density  of  Boltzmann  machine  relaxation.
real bm_relaxation_lpdf(vector x, matrix q, vector b){
     return sum(log_cosh(q*x+b)) -0.5*x'*x;
     }
}

data{
//  Number  of  dimension  in  Boltzmann  machine  binary  state.
int<lower=0> n_dim_b ;
//  Number  of  dimensions  in  relaxation  configuration  state.
int<lower=0> n_dim_r ;
//  Relaxation Q matrix  parameters .
matrix [n_dim_b,  n_dim_r]  q;
//  Relaxation  bias  vector  parameters.
vector[n_dim_b]  b ;
// Number of particles
int P;
//Fixed temperature
real beta;
}

parameters{
  vector[n_dim_r]  x[P]; //  Configuration  state.
}

model{
//  Set to target the pseudo-extended version of the Boltzmann machine relaxation log density.

  vector[P] num;
  vector[P] denom;
  vector[P] ratio;  
  real min_lr;

  for (p in 1:P) {
    num[p] = bm_relaxation_lpdf(x[p] | q, b);
    denom[p] = beta*num[p];
    }

  ratio = num - denom;
  min_lr = min(ratio);
  ratio = ratio - min_lr;
  target += log_sum_exp(ratio) + min_lr - log(P) + sum(denom);

}



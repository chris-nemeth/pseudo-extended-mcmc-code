//Standard Hamiltonian system for the Boltzmann relaxation example

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
}

parameters{
//  Configuration  state.
vector[n_dim_r]  x;
}

model{
//  Set to target  to  Boltzmann  machine relaxation log density.
x ~ bm_relaxation (q, b);
}

//Stan implementation of the banana target

data{
 real b; 
 real v; 
}

parameters{
  vector[2] x;
}

transformed parameters{
  vector[2] y;
  
  y = x;
  y[2] = x[2] - b*(x[1]^2-v);
  y[1] = x[1]/sqrt(v);
  
}

model{

  target += multi_normal_lpdf(y|rep_vector(0,2),diag_matrix(rep_vector(1.0,2)));
}
//Stan implementation of the flower target

data{
 real r;  // radius
 real A;  // amplitude
 real w;  // frequency
 real sigma; //standard deviation
}

parameters{
  vector[2] x;
}

model{
  real norm;
  real angle;
  real mu;

  norm = sqrt(x[1]^2+x[2]^2);
  angle = atan2(x[2],x[1]);
  mu = r + A*cos(w*angle);

  target += normal_lpdf(norm|mu,sigma);
}
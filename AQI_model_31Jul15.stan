functions {
  
  /** 
   * Transform standard normal to cauchy(location, scale).
   *
   * @param location Location parameter for Cauchy distribution
   * @param scale Scale parameter for Cauchy distribution
   * @param noise Standard normal variable (as declared in parameters block)
   */
  real cauchy_trans_lp(real location, real scale, real noise) {
    noise ~ normal(0,1) ;
    return location + scale * tan(pi() * (Phi_approx(noise) - 0.5)) ;
  }
  
  /** 
   * Transform standard normal to normal(location, scale).
   *
   * @param location Location parameter for normal distribution
   * @param scale Scale parameter for normal distribution
   * @param noise Standard normal variable (as declared in parameters block)
   */
  real normal_trans_lp(real location, real scale, real noise) {
    noise ~ normal(0,1) ;
    return location + scale * noise ;
  }
  
  /** 
   * Transform vector of standard normals to vector of normals
   * with same location (mean) and scale (sd).
   *
   * @param location Location for normal distributions
   * @param scale Scale for normal distributions
   * @param noise Name of vector of standard normals (as declared in parameters block)
   */
  vector normal_trans_vec_lp(real location, real scale, vector noise) {
    noise ~ normal(0,1) ;
    return location + scale * noise ;
  } 
}

data {
  // dimensions of data
  int<lower=1>                  nOb ;     // number of cases
  int<lower=1,upper=nOb>        nGrp1 ;    // number of groups (1) (facilities)
  int<lower=1,upper=nOb>        nGrp2 ;    // number of groups (2) (surgeries)
  int<lower=1>                  nX ;      // number of predictors
  
  // data 
  int<lower=0,upper=1>          y[nOb] ;   // outcome (antiemetic administration) 
  int<lower=1,upper=nGrp1>       g[nOb] ;   // map obs --> groups (cases --> facilities)
  int<lower=1,upper=nGrp2>       d[nOb] ;   // map obs --> groups (cases --> surgeries)
  row_vector[nX]                x[nOb] ;   // predictors (dummies for insurances status)

  // locations and scales for priors
  real                          alpha_loc ;       // alpha = intercept
  real                          beta_loc ;        // beta = fixed effects for predictors
  real                          sigma_gamma_loc ; // sigma_gamma = scale for gamma
  real                          sigma_delta_loc ; // sigma_delta = scale for delta
  real                          gamma_loc ;       // gamma = random effects for groups1
  real                          delta_loc ;       // delta = random effects for groups2
  real<lower=0>                 alpha_scale ;
  real<lower=0>                 beta_scale ;
  real<lower=0>                 sigma_gamma_scale ;
  real<lower=0>                 sigma_delta_scale ;

}

parameters {
  real              alpha_noise ;   
  vector[nX]        beta_noise ;    
  real<lower=0>     sigma_gamma_noise ;   
  real<lower=0>     sigma_delta_noise ;   

  vector[nGrp1]      gamma_noise ;
  vector[nGrp2]      delta_noise ;   
}

transformed parameters {
  real              alpha ;         // intercept
  vector[nX]        beta ;          // fixed effects for predictors
  real<lower=0>     sigma_gamma ;   // variability (for gammas)
  real<lower=0>     sigma_delta ;   // variability (for gammas)
  vector[nGrp1]      gamma ;         // random effects for groups1
  vector[nGrp2]      delta ;         // random effects for groups2

  alpha       <- normal_trans_lp(alpha_loc, alpha_scale, alpha_noise) ;
  beta        <- normal_trans_vec_lp(beta_loc, beta_scale, beta_noise) ;
  sigma_gamma <- cauchy_trans_lp(sigma_gamma_loc, sigma_gamma_scale, sigma_gamma_noise);
  sigma_delta <- cauchy_trans_lp(sigma_delta_loc, sigma_delta_scale, sigma_delta_noise) ;
  gamma       <- normal_trans_vec_lp(gamma_loc, sigma_gamma, gamma_noise) ;
  delta       <- normal_trans_vec_lp(delta_loc, sigma_delta, delta_noise) ;


  /* 
  Priors implied by transformations:
      alpha ~ normal(alpha_loc, alpha_scale)
      beta[i] ~ normal(beta_loc, beta_scale)
      sigma_gamma ~ half-cauchy(sigma_gamma_loc, sigma_gamma_scale)
      sigma_delta ~ half-cauchy(sigma_delta_loc, sigma_delta_scale)

      gamma[j] ~ normal(gamma_loc, sigma_gamma) 
      delta[j] ~ normal(delta_loc, delta_gamma) 

  */
}

model {
  vector[nOb] mu_logit ; // linear predictor

  for(n in 1:nOb) 
    mu_logit[n] <- alpha + x[n]*beta + gamma[g[n]] + delta[d[n]];
  
  y ~ bernoulli_logit(mu_logit) ; // vectorized 
}

/*
generated quantities {
  int y_rep[nOb] ;  // replicated data for posterior predictive checking
  
  for (n in 1:nOb)
    y_rep[n] <- bernoulli_rng(inv_logit(alpha + x[n]*beta + gamma[g[n]] + delta[d[n])) ;
}
*/
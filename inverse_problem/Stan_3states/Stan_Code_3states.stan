functions{
//function containing the ODEs
real [] pk_model(real t,
             real[] y,
             real[] theta,
             real[] rdata,
             int[] idata) {
  
  real dydt[3] ;
  real k21;real k12;real ke;
  
  k12 = exp(theta[1]);
  k21 = exp(theta[2]);
  ke = exp(theta[3]);
  
  dydt[1] = k21 * y[2] - k12 * y[1] - ke * y[1];
  dydt[2] =  k12 * y[1] - k21 * y[2];
  dydt[3] = ke * y[1];
 
  
  return dydt;       
  } 
}
//////////////////////////////////////////////////////////////////////////
  
  data{
    
    int<lower=0> N_subj;           // Number of individuals
    int<lower=0> N_param;          // Number of parameters
    int<lower=0> N_compart;        // Number of compartments
    int<lower=0>  N_obs;           // Total number of observations per intividual
    real<lower=0> time[N_obs];             // Time vector       
    real<lower=0> y0[N_compart];          // Initial concentration in compartments
    real t_init;                  // Initial time
    real eta_tr[N_param];         //prior parameter means
    real eta_tr_std[N_param];     //prior parameter standard deviatioin
    real a;                       // lkj_corr parameter
    matrix [N_obs,N_subj] central;       // Concentration of central
    matrix [N_obs,N_subj]peripheral;       // Concentration matrix
    matrix [N_obs,N_subj]excreta;       // Concentration matrix


    
  }

////////////////////////////////////////////////////////////////////////////
  
  transformed data {
    
    int idata[0];
    real rdata[0];
    vector[N_param]  eta_std ;              // Transformation of eta_tr in log-space
    vector[N_param]  eta ;                  // Transformation of eta_tr_std in log-space
    cov_matrix [N_param] H;                //covariance matrix
    
    for (i in 1:N_param){
      eta_std[i]= sqrt(log(((eta_tr_std[i]^2)/(eta_tr[i])^2)+1));
      eta[i]=log(((eta_tr[i])^2)/sqrt((eta_tr_std[i]^2)+(eta_tr[i])^2));
    }
    
    for (i in 1:N_param){
      for (j in 1:N_param){
        
        H[i,j]=0;
        
      }
    }
    
    H[1,1] = eta_std[1]^2;
    H[2,2] = eta_std[2]^2;
    H[3,3] = eta_std[3]^2;
    
  }
//////////////////////////////////////////////////////////////////
  
  parameters{
    vector[N_param] mu;               
    real<lower=0> sigma;               
    corr_matrix[N_param] C;
    vector<lower=0>[N_param] s; // scales
    vector [N_param] theta_tr[N_subj];    
  }
////////////////////////////////////////////////////
  transformed parameters{
    
    
  }
////////////////////////////////////////////////////////////////////
  
  model{
    
    real y_hat[N_obs,N_compart];
    matrix[N_param,N_param] Omega;
 
 
    //priors
    sigma ~ normal(0,1);
    mu ~ multi_normal(eta,H);
    C ~ lkj_corr(a); 
    s ~ normal(0,1);
    Omega = quad_form_diag(C, s);
    
    
    //likelihood~  
      
      for (j in 1:N_subj){

        theta_tr[j,:] ~ multi_normal(mu,Omega) ;
        y_hat[:,:] = integrate_ode_bdf(pk_model,y0,t_init,time,   to_array_1d(theta_tr[j,:]),
                                        rdata,idata);
        
        //log(central[:,j]) ~ normal(log( to_vector(y_hat[:,1])),sigma);
        //log(peripheral[:,j]) ~ normal(log( to_vector(y_hat[:,2])),sigma);
        //log(excreta[:,j]) ~ normal(log( to_vector(y_hat[:,3])),sigma);

        central[:,j] ~ normal(to_vector(y_hat[:,1]),sigma);
        peripheral[:,j] ~ normal(to_vector(y_hat[:,2]),sigma);
        excreta[:,j] ~ normal( to_vector(y_hat[:,3]),sigma);

      }
}

generated quantities{
  
  matrix[N_param,N_param] Omega;
  
  Omega = quad_form_diag(C, s);
}

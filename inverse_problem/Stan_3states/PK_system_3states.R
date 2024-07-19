library(rstan)

setwd('/Users/vassilis/Documents/GitHub/D-PINNs/inverse_problem/Stan_3states')


rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores(), stanc.allow_optimizations = TRUE,
        stanc.auto_format = TRUE
)

#load data
pk_data <- read.csv("/Users/vassilis/Documents/GitHub/D-PINNs/inverse_problem/train_data/stan_simulations_3states.csv")
time_vec <-  pk_data[,1]
central <-  pk_data[,2:21]
peripheral <-  pk_data[,22:41]
excreta <-  pk_data[,42:61]

#############################################################################################

pk_dat<-list(
  eta_tr = c(1,1,1), #CL, fKp, kp_AD
  eta_tr_std = c(1,1,1), 
  N_compart = 3, #Number of compartments
  N_subj = 20,		 #Number of individuals
  N_param = 3,    #Number of parameters(per individual)
  time = time_vec,
  central = central,
  peripheral = peripheral,
  excreta =  excreta,
  N_obs = dim(central)[1], #Total number of observations
  t_init = 0,             
  y0 = c(1,0,0),
  a=0.5
)

tic = proc.time()

fit <- stan(file = 'Stan_Code_3states.stan', data = pk_dat, 
            iter = 1000, warmup=400, chains=4)
tac = proc.time()
print(tac-tic)
options(max.print=5.5E5) 
print(fit, digits = 5)
#Check diagnostic tools
check_hmc_diagnostics(fit)

#Use shinystan for inference
library(shinystan)
#launch_shinystan(fit)

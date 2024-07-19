import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import csv
import copy

os.chdir('forward_problem')

# Define the Lorenz system ODEs
def PK(t, y, k12, k21, ke):
    dydt = [k21 * y[1] - k12 * y[0] - ke * y[0],
            k12 * y[0] - k21 * y[1],
            ke * y[0]
        ]
    return dydt

# Define the mean values of the system's parameters
k12_mean = 0.5
k21_mean = 0.2
ke_mean = 0.3
# Estimate the standard deviation  of these parameters based on CV = sd/mean
#CV = 0.3
sd_k12 = (0.2 * k12_mean)
sd_k21 = (0.2 * k21_mean)
sd_ke = (0.1 * ke_mean)

mu_k12 = np.log((k12_mean**2) / np.sqrt((k12_mean**2) + (sd_k12**2)))
sigma_12 =  np.sqrt(np.log(1 + ((sd_k12**2) / (k12_mean**2))))

mu_k21 = np.log((k21_mean**2) / np.sqrt((k21_mean**2) + (sd_k21**2)))
sigma_21 =  np.sqrt(np.log(1 + ((sd_k21**2) / (k21_mean**2))))

mu_ke = np.log((ke_mean**2) / np.sqrt((ke_mean**2) + (sd_ke**2)))
sigma_ke =  np.sqrt(np.log(1 + ((sd_ke**2) / (ke_mean**2))))

# Sample N values from the parameters' distributions to produce the data
N_samples = 1000

mu = np.array([mu_k12, mu_k21, mu_ke])
S = np.diag([sigma_12, sigma_21, sigma_ke])

cor_matrix = np.matrix([[1, -0.4, -0.7],
                        [-0.4, 1, 0.2],
                        [-0.7, 0.2, 1]])

cov_matrix = np.matmul(np.matmul(S, cor_matrix), S)
np.random.seed(113)
mvn_samples = np.random.multivariate_normal(mu, cov_matrix, N_samples)

k12_samples = np.exp(mvn_samples[:,0])
k21_samples = np.exp(mvn_samples[:,1])
ke_samples = np.exp(mvn_samples[:,2])

# Create a dataframe with the samples of the parameters
params = pd.DataFrame({'k12': k12_samples, 'k21': k21_samples, 'ke': ke_samples})
doses = np.linspace(1, 50, 25)

for i in range(len(doses)):
    dose = doses[i]
    # Set the initial conditions
    y0 = [dose, 0.0, 0.0]

    # Set the time span
    t_span = (0, 5.0)
    t_eval = np.linspace(t_span[0], t_span[1], 25)

    # Solve the ODEs using solve_ivp
    preds_x = np.zeros((len(t_eval),1))
    preds_y = np.zeros((len(t_eval),1))
    preds_z = np.zeros((len(t_eval),1))
    for j in range(N_samples):
        k12 = params['k12'][j]
        k21 = params['k21'][j]
        ke = params['ke'][j]
        sol = solve_ivp(PK, t_span, y0, args=(k12, k21, ke),t_eval=t_eval)

        preds_x = np.hstack((preds_x, np.vstack(sol.y[0]) + np.random.normal(0, 0.001, size=(len(t_eval), 1)) )) 
        preds_y = np.hstack((preds_y, np.vstack(sol.y[1]) + np.random.normal(0, 0.001, size=(len(t_eval), 1)) )) 
        preds_z = np.hstack((preds_z, np.vstack(sol.y[2]) + np.random.normal(0, 0.001, size=(len(t_eval), 1)) )) 
    # Estimate the mean and the variance of each output variable
    x_mean = np.mean(preds_x[:,1:], axis=1)
    x_var = np.var(preds_x[:,1:], axis=1)
    y_mean = np.mean(preds_y[:,1:], axis=1)
    y_var = np.var(preds_y[:,1:], axis=1)
    z_mean = np.mean(preds_z[:,1:], axis=1)
    z_var = np.var(preds_z[:,1:], axis=1)
    if i == 0:
        # Create DataFrame in the first iteration
        df = pd.DataFrame({'dose': np.full(t_eval.shape[0], dose), 't': t_eval[:], 'X': x_mean, 'X_var':x_var,
                   'Y': y_mean, 'Y_var': y_var,
                   'Z': z_mean, 'Z_var': z_var})
    else:

        # Append data to the existing DataFrame in subsequent iterations
        new_data = pd.DataFrame({'dose':np.full( t_eval.shape[0], dose) , 't': t_eval[:], 'X': x_mean, 'X_var':x_var,
                   'Y': y_mean, 'Y_var': y_var,
                   'Z': z_mean, 'Z_var': z_var})
        df = pd.concat([df, new_data], ignore_index=True)

states_corr_matrix = np.corrcoef((preds_x[4,1:], preds_y[4,1:], preds_z[4,1:], k12_samples, k21_samples, ke_samples ))
print(states_corr_matrix)

mu_k12_obs = np.log((np.mean(k12_samples)**2) / np.sqrt((np.mean(k12_samples)**2) + (np.var(k12_samples))))
sigma_12_obs =  np.sqrt(np.log(1 + ((np.var(k12_samples)) / (np.mean(k12_samples)**2))))
print(mu_k12_obs)
print(sigma_12_obs)

mu_k21_obs = np.log((np.mean(k21_samples)**2) / np.sqrt((np.mean(k21_samples)**2) + (np.var(k21_samples))))
sigma_21_obs =  np.sqrt(np.log(1 + ((np.var(k21_samples)) / (np.mean(k21_samples)**2))))
print(mu_k21_obs)
print(sigma_21_obs)

mu_ke_obs = np.log((np.mean(ke_samples)**2) / np.sqrt((np.mean(ke_samples)**2) + (np.var(ke_samples))))
sigma_e_obs =  np.sqrt(np.log(1 + ((np.var(ke_samples)) / (np.mean(ke_samples)**2))))
print(mu_ke_obs)
print(sigma_e_obs)

with open('train_data/parameters_for_sampling.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([ 'k12', 'k12_sd', 'k21', 'k21_sd', 'ke',  'ke_sd',
                     'c_x_y', 'c_x_z', 'c_x_k12', 'c_x_k21',    'c_x_ke',
                     'c_y_z',    'c_y_k12', 'c_y_k21', 'c_y_ke',
                     'c_z_k12', 'c_z_k21', 'c_z_ke',
                      'c_k12_k21',    'c_k12_ke', 'c_k21_ke'])
    writer.writerow([mu_k12_obs, sigma_12_obs, mu_k21_obs, sigma_21_obs, mu_ke_obs, sigma_e_obs,
                     states_corr_matrix[0,1] ,  states_corr_matrix[0,2],states_corr_matrix[0,3], states_corr_matrix[0,4], states_corr_matrix[0,5], 
                     states_corr_matrix[1,2],states_corr_matrix[1,3], states_corr_matrix[1,4], states_corr_matrix[1,5],
                     states_corr_matrix[2,3], states_corr_matrix[2,4], states_corr_matrix[2,5], 
                     states_corr_matrix[3,4], states_corr_matrix[3,5],
                     states_corr_matrix[4,5]
                     ])
# Select scaling method from 'log', 'min_max' or 'log_min_max'
method = 'no_scale' 

if method== 'log':
    df_non_zero =   copy.deepcopy(df)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
          if df_non_zero.iloc[i,j] <=0:
            df_non_zero.iloc[i,j] = df_non_zero.iloc[i+1,j]/1000
    df_tr = np.log(df_non_zero)
    df_tr.to_csv('train_data/PK_simulation_forward.csv', index=False)
elif method == 'log_min_max':
    # Create a new dataframe that contains non-zero values
    df_non_zero =   copy.deepcopy(df)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
          if df_non_zero.iloc[i,j] <=0:
            df_non_zero.iloc[i,j] = df_non_zero.iloc[i+1,j]/1000
    df_tr = np.log(df_non_zero)
    dose_max = np.max(df_tr['dose'])
    dose_min = np.min(df_tr['dose'])     
    t_max = np.max(df_tr['t'])
    t_min = np.min(df_tr['t'])
    x_mean_max = np.max(df_tr['X'])
    x_mean_min = np.min(df_tr['X'])
    y_mean_max = np.max(df_tr['Y'])
    y_mean_min = np.min(df_tr['Y'])
    z_mean_max = np.max(df_tr['Z'])
    z_mean_min = np.min(df_tr['Z'])
    x_var_max = np.max(df_tr['X_var'])
    x_var_min = np.min(df_tr['X_var'])
    y_var_max = np.max(df_tr['Y_var'])
    y_var_min = np.min(df_tr['Y_var'])
    z_var_max = np.max(df_tr['Z_var'])
    z_var_min = np.min(df_tr['Z_var'])
    scaler = MinMaxScaler()#StandardScaler()
    scaler.fit(df_tr.iloc[:, :])
    df_tr = pd.DataFrame(scaler.transform(df_tr.iloc[:, :]),  columns=df.columns)
    df_tr.to_csv('train_data/PK_simulation_forward.csv', index=False)
elif method == 'min_max':
    df_non_neg =   copy.deepcopy(df)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
          if df_non_neg.iloc[i,j] <0:
            df_non_neg.iloc[i,j] = 0
    dose_max = np.max(df_non_neg['dose'])
    dose_min = np.min(df_non_neg['dose'])       
    t_max = np.max(df_non_neg['t'])
    t_min = np.min(df_non_neg['t'])
    x_mean_max = np.max(df_non_neg['X'])
    x_mean_min = np.min(df_non_neg['X'])
    y_mean_max = np.max(df_non_neg['Y'])
    y_mean_min = np.min(df_non_neg['Y'])
    z_mean_max = np.max(df_non_neg['Z'])
    z_mean_min = np.min(df_non_neg['Z'])
    x_var_max = np.max(df_non_neg['X_var'])
    x_var_min = np.min(df_non_neg['X_var'])
    y_var_max = np.max(df_non_neg['Y_var'])
    y_var_min = np.min(df_non_neg['Y_var'])
    z_var_max = np.max(df_non_neg['Z_var'])
    z_var_min = np.min(df_non_neg['Z_var'])
    scaler = MinMaxScaler()#StandardScaler()
    scaler.fit(df_non_neg.iloc[:, :])
    df_non_neg = pd.DataFrame(scaler.transform(df_non_neg.iloc[:, :]),  columns=df.columns)
    df_non_neg.to_csv('train_data/PK_simulation_forward.csv', index=False)
elif method == 'min_max_no_dosetime':
    df_non_neg =   copy.deepcopy(df)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
          if df_non_neg.iloc[i,j] <0:
            df_non_neg.iloc[i,j] = 0
    dose_max = np.max(df_non_neg['dose'])
    dose_min = np.min(df_non_neg['dose'])       
    t_max = np.max(df_non_neg['t'])
    t_min = np.min(df_non_neg['t'])
    x_mean_max = np.max(df_non_neg['X'])
    x_mean_min = np.min(df_non_neg['X'])
    y_mean_max = np.max(df_non_neg['Y'])
    y_mean_min = np.min(df_non_neg['Y'])
    z_mean_max = np.max(df_non_neg['Z'])
    z_mean_min = np.min(df_non_neg['Z'])
    x_var_max = np.max(df_non_neg['X_var'])
    x_var_min = np.min(df_non_neg['X_var'])
    y_var_max = np.max(df_non_neg['Y_var'])
    y_var_min = np.min(df_non_neg['Y_var'])
    z_var_max = np.max(df_non_neg['Z_var'])
    z_var_min = np.min(df_non_neg['Z_var'])
    scaler = MinMaxScaler()#StandardScaler()
    scaler.fit(df_non_neg.iloc[:, 2:])
    df_non_neg[:, 2:] = pd.DataFrame(scaler.transform(df_non_neg.iloc[:, 2:]),  columns=df.columns)
    df_non_neg.to_csv('train_data/PK_simulation_forward.csv', index=False)
elif method == 'no_scale':
    dose_max = np.max(df['dose'])
    dose_min = np.min(df['dose'])     
    t_max = np.max(df['t'])
    t_min = np.min(df['t'])
    x_mean_max = np.max(df['X'])
    x_mean_min = np.min(df['X'])
    y_mean_max = np.max(df['Y'])
    y_mean_min = np.min(df['Y'])
    z_mean_max = np.max(df['Z'])
    z_mean_min = np.min(df['Z'])
    x_var_max = np.max(df['X_var'])
    x_var_min = np.min(df['X_var'])
    y_var_max = np.max(df['Y_var'])
    y_var_min = np.min(df['Y_var'])
    z_var_max = np.max(df['Z_var'])
    z_var_min = np.min(df['Z_var'])
    df.to_csv('train_data/PK_simulation_forward.csv', index=False)
    
with open('train_data/parameters_forward.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['dose_max', 'dose_min', 't_max', 't_min', 'x_mean_max', 
                    'x_mean_min', 'y_mean_max', 'y_mean_min', 
                    'z_mean_max', 'z_mean_min',
                    'x_var_max',  'x_var_min', 
                    'y_var_max', 'y_var_min', 
                    'z_var_max', 'z_var_min'])
    writer.writerow([dose_max, dose_min, t_max, t_min, x_mean_max, 
                    x_mean_min, y_mean_max, y_mean_min, 
                    z_mean_max, z_mean_min,
                    x_var_max,  x_var_min, 
                    y_var_max, y_var_min, 
                    z_var_max, z_var_min])

###################################################
#--------------------------------------------------
# Redo the analysis for three new doses to plot    
#--------------------------------------------------
###################################################    

doses = np.array([6.5,27.3, 46.7])

for i in range(len(doses)):
    dose = doses[i]
    # Set the initial conditions
    y0 = [dose, 0.0, 0.0]

    # Set the time span
    t_span = (0, 5.0)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Solve the ODEs using solve_ivp
    preds_x = np.zeros((len(t_eval),1))
    preds_y = np.zeros((len(t_eval),1))
    preds_z = np.zeros((len(t_eval),1))
    for j in range(N_samples):
        k12 = params['k12'][j]
        k21 = params['k21'][j]
        ke = params['ke'][j]
        sol = solve_ivp(PK, t_span, y0, args=(k12, k21, ke),
                        t_eval=t_eval)

        preds_x = np.hstack((preds_x, np.vstack(sol.y[0]) + np.random.normal(0, 0.001, size=(len(t_eval), 1)) )) 
        preds_y = np.hstack((preds_y, np.vstack(sol.y[1]) + np.random.normal(0, 0.001, size=(len(t_eval), 1)) )) 
        preds_z = np.hstack((preds_z, np.vstack(sol.y[2]) + np.random.normal(0, 0.001, size=(len(t_eval), 1)) )) 
    # Estimate the mean and the variance of each output variable
    x_mean = np.mean(preds_x[:,1:], axis=1)
    x_var = np.var(preds_x[:,1:], axis=1)
    y_mean = np.mean(preds_y[:,1:], axis=1)
    y_var = np.var(preds_y[:,1:], axis=1)
    z_mean = np.mean(preds_z[:,1:], axis=1)
    z_var = np.var(preds_z[:,1:], axis=1)
    if i == 0:
        # Create DataFrame in the first iteration
        df_pred = pd.DataFrame({'dose': np.full(t_eval.shape[0], dose), 't': t_eval[:], 'X': x_mean, 'X_var':x_var,
                   'Y': y_mean, 'Y_var': y_var,
                   'Z': z_mean, 'Z_var': z_var})
    else:

        # Append data to the existing DataFrame in subsequent iterations
        new_data = pd.DataFrame({'dose':np.full( t_eval.shape[0], dose) , 't': t_eval[:], 'X': x_mean, 'X_var':x_var,
                   'Y': y_mean, 'Y_var': y_var,
                   'Z': z_mean, 'Z_var': z_var})
        df_pred = pd.concat([df_pred, new_data], ignore_index=True)

if method == 'log':
    df_pred_non_zero =   copy.deepcopy(df_pred)
    for i in range(df_pred.shape[0]):
        for j in range(df_pred.shape[1]):
          if df_pred_non_zero.iloc[i,j] <=0:
            df_pred_non_zero.iloc[i,j] = df_pred_non_zero.iloc[i+1,j]/1000
    df_pred_tr = np.log(df_pred_non_zero)
    df_pred_tr.to_csv('train_data/ODE_samples.csv', index=False)
elif method == 'log_min_max':
    # Create a new dataframe that contains non-zero values
    df_pred_non_zero =   copy.deepcopy(df_pred)
    for i in range(df_pred.shape[0]):
        for j in range(df_pred.shape[1]):
          if df_pred_non_zero.iloc[i,j] <=0:
            df_pred_non_zero.iloc[i,j] = df_pred_non_zero.iloc[i+1,j]/1000
    df_pred_tr = np.log(df_pred_non_zero)
    df_pred_tr = pd.DataFrame(scaler.transform(df_pred_tr.iloc[:, :]),  columns=df_pred.columns)
    df_pred_tr.to_csv('train_data/ODE_samples.csv', index=False)
elif method == 'min_max':
    df_pred_non_neg =   copy.deepcopy(df_pred)
    for i in range(df_pred.shape[0]):
        for j in range(df_pred.shape[1]):
          if df_pred_non_neg.iloc[i,j] <0:
            df_pred_non_neg.iloc[i,j] = 0
    df_pred_non_neg = pd.DataFrame(scaler.transform(df_pred_non_neg.iloc[:, :]),  columns=df_pred.columns)
    df_pred_non_neg.to_csv('train_data/ODE_samples.csv', index=False)
elif method == 'min_max_no_dosetime':
    df_pred_non_neg =   copy.deepcopy(df)
    for i in range(df_pred.shape[0]):
        for j in range(df_pred.shape[1]):
            if df_pred_non_neg.iloc[i,j] <0:
                df_pred_non_neg.iloc[i,j] = 0
    df_pred_non_neg[:, 2:] = pd.DataFrame(scaler.transform(df_pred_non_neg.iloc[:, 2:]),  columns=df_pred.columns)
    df_pred_non_neg.to_csv('train_data/ODE_samples.csv', index=False)
elif method == 'no_scale':
    df_pred.to_csv('train_data/ODE_samples.csv', index=False)
    



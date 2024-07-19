import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import csv
import time


os.chdir('inverse_problem')

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
CV = 0.2
sd_k12 = (CV * k12_mean)
sd_k21 = (CV * k21_mean)
sd_ke = (CV * ke_mean)

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
# Set the initial conditions
y0 = [1.0, 0.0, 0.0]

# Set the time span
t_span = (0, 5.0)
t_eval = np.linspace(t_span[0], t_span[1], 10)

# Solve the ODEs using solve_ivp
preds_x = np.zeros((len(t_eval),1))
preds_y = np.zeros((len(t_eval),1))
preds_z = np.zeros((len(t_eval),1))
# Record the start time
start_time = time.time()

for i in range(N_samples):
    k12 = params['k12'][i]
    k21 = params['k21'][i]
    ke = params['ke'][i]
    sol = solve_ivp(PK, t_span, y0, args=(k12, k21, ke),
                     t_eval=t_eval)

    preds_x = np.hstack((preds_x, np.vstack(sol.y[0]) + np.random.normal(0, 0.001, size=(len(t_eval), 1)) )) 
    preds_y = np.hstack((preds_y, np.vstack(sol.y[1]) + np.random.normal(0, 0.001, size=(len(t_eval), 1)) )) 
    preds_z = np.hstack((preds_z, np.vstack(sol.y[2]) + np.random.normal(0, 0.001, size=(len(t_eval), 1)) )) 
    #print(i)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time} seconds")

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

# Save the noisy data in a csv file
stan_df = pd.DataFrame({'t': t_eval[1:], 'x_1': preds_x[1:,1], 'x_2': preds_x[1:,2],'x_3': preds_x[1:,3], 'x_4': preds_x[1:,4],
                   'x_5': preds_x[1:,5], 'x_6': preds_x[1:,6],'x_7': preds_x[1:,7], 'x_8': preds_x[1:,8],
                   'x_9': preds_x[1:,9], 'x_10': preds_x[1:,10],'x_11': preds_x[1:,11], 'x_12': preds_x[1:,12],
                   'x_13': preds_x[1:,13], 'x_14': preds_x[1:,14],'x_15': preds_x[1:,15], 'x_16': preds_x[1:,16],
                   'x_17': preds_x[1:,17], 'x_18': preds_x[1:,18],'x_19': preds_x[1:,19], 'x_20': preds_x[1:,20],

                   'y_1': preds_y[1:,1], 'y_2': preds_y[1:,2],'y_3': preds_y[1:,3], 'y_4': preds_y[1:,4],
                   'y_5': preds_y[1:,5], 'y_6': preds_y[1:,6],'y_7': preds_y[1:,7], 'y_8': preds_y[1:,8],
                   'y_9': preds_y[1:,9], 'y_10': preds_y[1:,10],'y_11': preds_y[1:,11], 'y_12': preds_y[1:,12],
                   'y_13': preds_y[1:,13], 'y_14': preds_y[1:,14],'y_15': preds_y[1:,15], 'y_16': preds_y[1:,16],
                   'y_17': preds_y[1:,17], 'y_18': preds_y[1:,18],'y_19': preds_y[1:,19], 'y_20': preds_y[1:,20],

                   'z_1': preds_z[1:,1], 'z_2': preds_z[1:,2],'z_3': preds_z[1:,3], 'z_4': preds_z[1:,4],
                   'z_5': preds_z[1:,5], 'z_6': preds_z[1:,6],'z_7': preds_z[1:,7], 'z_8': preds_z[1:,8],
                   'z_9': preds_z[1:,9], 'z_10': preds_z[1:,10],'z_11': preds_z[1:,11], 'z_12': preds_z[1:,12],
                   'z_13': preds_z[1:,13], 'z_14': preds_z[1:,14],'z_15': preds_z[1:,15], 'z_16': preds_z[1:,16],
                   'z_17': preds_z[1:,17], 'z_18': preds_z[1:,18],'z_19': preds_z[1:,19], 'z_20': preds_z[1:,20],
                   })
stan_df.to_csv('train_data/stan_simulations_3states.csv', index=False)

# Estimate the mean and the variance of each output variable
x_mean = np.mean(preds_x[:,1:], axis=1)
x_var = np.var(preds_x[:,1:], axis=1)
y_mean = np.mean(preds_y[:,1:], axis=1)
y_var = np.var(preds_y[:,1:], axis=1)
z_mean = np.mean(preds_z[:,1:], axis=1)
z_var = np.var(preds_z[:,1:], axis=1)
x_upper_quantile = np.percentile(preds_x[:,1:], 97.5 , axis=1)
x_lower_quantile = np.percentile(preds_x[:,1:], 2.5 , axis=1)
y_upper_quantile = np.percentile(preds_y[:,1:], 97.5 , axis=1)
y_lower_quantile = np.percentile(preds_y[:,1:], 2.5 , axis=1)
z_upper_quantile = np.percentile(preds_z[:,1:], 97.5 , axis=1)
z_lower_quantile = np.percentile(preds_z[:,1:], 2.5 , axis=1)

# Save the noisy data in a csv file
df = pd.DataFrame({'t': t_eval[:], 'X': x_mean, 'X_var':x_var,
                   'Y': y_mean, 'Y_var': y_var,
                   'Z': z_mean, 'Z_var': z_var})


scaler = MinMaxScaler()#StandardScaler()
df.iloc[:, 1:] = scaler.fit_transform(df.iloc[:, 1:])
t_bar = np.mean(t_eval)
t_sd = np.std(t_eval)
x_mean_bar = np.mean(x_mean)
x_mean_sd = np.std(x_mean)
y_mean_bar = np.mean(y_mean)
y_mean_sd = np.std(y_mean)
z_mean_bar = np.mean(z_mean)
z_mean_sd = np.std(z_mean)
x_var_bar = np.mean(x_var)
x_var_sd = np.std(x_var)
y_var_bar = np.mean(y_var)
y_var_sd = np.std(y_var)
z_var_bar = np.mean(z_var)
z_var_sd = np.std(z_var)

t_max = np.max(t_eval)
t_min = np.min(t_eval)
x_mean_max = np.max(x_mean)
x_mean_min = np.min(x_mean)
y_mean_max = np.max(y_mean)
y_mean_min = np.min(y_mean)
z_mean_max = np.max(z_mean)
z_mean_min = np.min(z_mean)
x_var_max = np.max(x_var)
x_var_min = np.min(x_var)
y_var_max = np.max(y_var)
y_var_min = np.min(y_var)
z_var_max = np.max(z_var)
z_var_min = np.min(z_var)

# Estimate the mean and the variance of each output variable
x_max = np.max(preds_x[:,1:])
x_min = np.min(preds_x[:,1:])
y_max = np.max(preds_y[:,1:])
y_min = np.min(preds_y[:,1:])
z_max = np.max(preds_z[:,1:])
z_min = np.min(preds_z[:,1:])

df.to_csv('PK_simulation.csv', index=False)

with open('parameters.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['t_bar', 't_sd','x_mean_bar', 'x_mean_sd', 'y_mean_bar', 'y_mean_sd', 
                    'z_mean_bar', 'z_mean_sd', 
                    'x_var_bar', 'x_var_sd', 'y_var_bar', 'y_var_sd',
                    'z_var_bar', 'z_var_sd',
                    't_max', 't_min', 'x_mean_max', 
                    'x_mean_min', 'y_mean_max', 'y_mean_min', 
                    'z_mean_max', 'z_mean_min',
                    'x_var_max',  'x_var_min', 
                    'y_var_max', 'y_var_min', 
                    'z_var_max', 'z_var_min',
                    "x_max", "x_min", "y_max", "y_min",
                    "z_max", "z_min"])
    writer.writerow([t_bar, t_sd,x_mean_bar, x_mean_sd, y_mean_bar, y_mean_sd, 
                    z_mean_bar, z_mean_sd, 
                    x_var_bar, x_var_sd, y_var_bar, y_var_sd,
                    z_var_bar, z_var_sd,
                    t_max, t_min, x_mean_max, 
                    x_mean_min, y_mean_max, y_mean_min, 
                    z_mean_max, z_mean_min,
                    x_var_max,  x_var_min, 
                    y_var_max, y_var_min, 
                    z_var_max, z_var_min,
                    x_max, x_min, y_max, y_min,
                    z_max, z_min])

# Plot the results

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot()
ax.plot(t_eval, x_mean, lw=2, label = 'x')
ax.plot(t_eval, y_mean, lw=2, label = 'y')
ax.plot(t_eval, z_mean, lw=2, label = 'z')
ax.plot(t_eval, x_upper_quantile, 'r--', label = 'x Upper 95% quantile')
ax.plot(t_eval, x_lower_quantile, 'r--', label = 'x lower 95% quantile')
ax.plot(t_eval, y_upper_quantile, 'r--', label = 'y Upper 95% quantile')
ax.plot(t_eval, y_lower_quantile, 'r--', label = 'y lower 95% quantile')
ax.plot(t_eval, z_upper_quantile, 'r--', label = 'z Upper 95% quantile')
ax.plot(t_eval, z_lower_quantile, 'r--', label = 'z lower 95% quantile')
ax.set_title('Pharmacokinetic System Simulation')
ax.legend()
plt.show()

# Set the time span
# t_span = (0, 10)
# t_eval = np.linspace(t_span[0], t_span[1], 100)

# y0 = [1.0, 0.0]
# k12 = 0.3
# k21 = 0.2
# ke = 0.3

# sol = solve_ivp(PK, t_span, y0, args=(k12, k21, ke),
#                      t_eval=t_eval)

# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot()
# ax.plot(t_eval, sol.y[0], lw=2, label = 'Cc')
# ax.plot(t_eval, sol.y[1], lw=2, label = 'Cp')
# ax.set_title('Lotka-Voltera System Simulation')
# ax.legend()
# plt.show()
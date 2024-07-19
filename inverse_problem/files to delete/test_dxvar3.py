import deepxde as dde
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import os
import csv
import matplotlib.pyplot as plt

os.chdir('/Users/vassilis/Documents/GitHub/Lotka_Voltera/PK_PINN')

# Import the parameters relevant to the scaling of the data
with open('parameters.csv', 'r') as csvfile:
    my_reader = csv.reader(csvfile)
    header = next(my_reader)  # Skip the header row
    data_row = next(my_reader)  # Read the data row

# Assign values to variables
t_bar, t_sd, x_mean_bar,x_mean_sd,y_mean_bar,y_mean_sd,x_var_bar,x_var_sd,y_var_bar,\
y_var_sd,t_max, t_min, x_mean_max, x_mean_min, y_mean_max, y_mean_min, x_var_max,\
x_var_min, y_var_max, y_var_min, x_max, x_min, y_max, y_min  = map(float, data_row)

def gen_traindata(file_name):
    data = pd.read_csv(file_name)
    return np.vstack(data["t"]) , np.vstack(data["X"]), np.vstack(data["X_var"]), np.vstack(data["Y"]), np.vstack(data["Y_var"])

t_obs, x_obs, x_var_obs, y_obs, y_var_obs = gen_traindata('PK_simulation.csv')
nn_preds = model.predict(t_obs)
# Repeat the pattern to create the desired array
array1 = np.tile(0.0, (4, 1))
array2 = np.tile(t_obs, (4, 1))
array3 = np.array([0.0,5.0])
array4 = np.random.uniform(0, 5, 100)


k12 = -0.7
k12_sd = -1.6
x_mean = nn_preds[:, 0:1]*(x_mean_max-x_mean_min)+x_mean_min #*x_mean_sd + x_mean_bar
x_var =  nn_preds[:, 1:2]*(x_var_max-x_var_min)+x_var_min#*x_var_sd+x_var_bar
y_mean = nn_preds[:, 2:3]*(y_mean_max-y_mean_min)+y_mean_min#y_mean_sd + y_mean_bar, 
y_var = nn_preds[:, 3:4]*(y_var_max-y_var_min)+y_var_min#y_var_sd+y_var_bar     

# Sample N values from x,y
N = 1000000
Seed = 12322921
eps = 1e-15
x_mean += eps
y_mean += eps
x_var = tf.math.abs(x_var)
y_var = tf.math.abs(y_var)
N_par=3
mu_x = tf.math.log((x_mean**2) / tf.math.sqrt((x_mean**2) + (x_var)))
sigma_x = tf.math.sqrt(tf.math.log(1 + ((x_var) / (x_mean**2))))
mu_y = tf.math.log((y_mean**2) / tf.math.sqrt((y_mean**2) + (y_var)))
sigma_y = tf.math.sqrt(tf.math.log(1 + ((y_var) / (y_mean**2))))

corr_matrix = tf.constant([[1.0,-1.0, -1.0], [-1.0,1.0, 1.0], [-1.0, 1.0, 1.0]])
size = tf.shape(mu_x)[0]
mu = tf.concat([mu_x, mu_y, tf.fill(dims=[size,1], value=k12)], axis = 1)
S = tf.linalg.diag(tf.concat([sigma_x, sigma_y, tf.math.exp(tf.fill(dims=[size,1],
                                            value = k12_sd))],axis = 1))
cov_mat = tf.linalg.matmul(tf.linalg.matmul(S,corr_matrix),S)
# Add a small number to the diagonal to make the covariance matrix positive definite
cov_mat += 1e-6 * tf.eye(N_par)

tf.random.set_seed(seed=Seed)
mvn = tfp.distributions.MultivariateNormalFullCovariance(mu, cov_mat).sample(N)
tf.random.set_seed(seed=Seed)
x_samples = tf.transpose(tf.exp(mvn[:,:,0]))
#x_samples_scaled = (x_samples - x_min)/(x_max - x_min) 
y_samples = tf.transpose(tf.exp(mvn[:,:,1]))
#y_samples_scaled = (y_samples - y_min)/(y_max - y_min)
k12_samples =  tf.transpose(tf.exp(mvn[:,:,2]))
# k12_samples =  tf.math.reduce_mean(tf.transpose(tf.exp(mvn[:,:,2])), axis=0, keepdims=True)

k21 = 0.2 
ke = 0.3 
x_bar =  tf.math.reduce_mean(x_samples, axis=1, keepdims=True)
y_bar =  tf.math.reduce_mean(y_samples, axis=1, keepdims=True)

dx_rs_unscaled = k21 * y_samples - k12_samples * x_samples - ke * x_samples
dy_rs_unscaled = k12_samples * x_samples - k21 * y_samples

dx_rs_mean_unscaled = tf.math.reduce_mean(dx_rs_unscaled, axis=1, keepdims=True)
dx_rs_mean_scaled = dx_rs_mean_unscaled / (x_mean_max - x_mean_min)

dy_rs_mean_unscaled = tf.math.reduce_mean(dy_rs_unscaled, axis=1, keepdims=True)
dy_rs_mean_scaled = dy_rs_mean_unscaled/(y_mean_max - y_mean_min)

dx_rs_var_unscaled =  (2/(N-1)) *tf.math.reduce_sum((x_samples - x_bar) *\
                                (dx_rs_unscaled  - dx_rs_mean_unscaled), axis=1, keepdims=True)
dx_rs_var_scaled =  dx_rs_var_unscaled / (x_var_max-x_var_min) 

dy_rs_var_unscaled =  (2/(N-1)) * tf.math.reduce_sum((y_samples - y_bar) *\
                                (dy_rs_unscaled - dy_rs_mean_unscaled), axis=1, keepdims=True)
dy_rs_var_scaled =  dy_rs_var_unscaled / (y_var_max-y_var_min) 


print(dx_rs_var_scaled)

a = tf.Variable(5.0)

ph = tf.compat.v1.placeholder(tf.float32)
assign_op = tf.assign(ph, a )




# Define a lambda function that takes a tf.Variable() and returns a tf.constant()
lambda_func = lambda v: tf.constant(v)

# Use the lambda function to create a tf.constant() tensor
constant_tensor = lambda_func(a)

corr_matrix = tf.constant([[1.0,-1.0, -1.0], [-1.0,1.0, 1.0], [-1.0, 1.0, 1.0]])
# Assign the tf.Variable() tensor to the placeholder
assign_op = tf.assign(corr_matrix, corr_matrix)


# Create a variable tensor
var1 = tf.Variable([1], dtype=tf.float32)
var2 = tf.Variable([2], dtype=tf.float32)
var3 = tf.Variable([3], dtype=tf.float32)
var4 = tf.Variable([4], dtype=tf.float32)
var5 = tf.Variable([5], dtype=tf.float32)
var6 = tf.Variable([6], dtype=tf.float32)

# Define a lambda function that takes a tf.Variable() and returns a tf.constant()
tanh = lambda z:  ((tf.math.exp(1.0) **z - tf.math.exp(1.0) **(-z)) / (tf.math.exp(1.0) **z + tf.math.exp(1.0) **(-z)) )

combo1 = tf.stack([tanh(var1), tanh(var2), tanh(var3)])
combo2 = tf.stack([var4, var5, var6])
combo3 = tf.stack([var4, var5, var6])
# Stack the variables into a constant tensor
not_trans = tf.constant(tf.concat([combo1, combo2, combo3],1 ) )
const = tf.transpose(not_trans )
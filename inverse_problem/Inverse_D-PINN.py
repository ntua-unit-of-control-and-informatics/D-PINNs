import deepxde as dde
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import os
import csv
import matplotlib.pyplot as plt

# # #Print only the total loss
from deepxde.utils import list_to_str
from deepxde.display import TrainingDisplay

def custom_call(self, train_state):
    if not self.is_header_print:
        self.len_train = len(train_state.loss_train) * 2 + 1
        self.len_test = len(train_state.loss_test) * 2 + 1
        self.len_metric = len(train_state.metrics_test) * 2 + 1
        self.header()
    self.print_one(
        str(train_state.step),
        list_to_str(np.sum(train_state.loss_train)), 
        list_to_str(np.sum(train_state.loss_test)),
        list_to_str(train_state.metrics_test),
    )

TrainingDisplay.__call__ = custom_call

# Add gradient clipping and print their values
from deepxde import Model, config, optimizers

def custom_compile_tensorflow(self, lr, loss_fn, decay, loss_weights):
        """tensorflow"""

        @tf.function(jit_compile=config.xla_jit)
        def outputs(training, inputs):
            return self.net(inputs, training=training)

        def outputs_losses(training, inputs, targets, auxiliary_vars, losses_fn):
            self.net.auxiliary_vars = auxiliary_vars
            # Don't call outputs() decorated by @tf.function above, otherwise the
            # gradient of outputs wrt inputs will be lost here.
            outputs_ = self.net(inputs, training=training)
            # Data losses
            losses = losses_fn(targets, outputs_, loss_fn, inputs, self)
            if not isinstance(losses, list):
                losses = [losses]
            # Regularization loss
            if self.net.regularizer is not None:
                losses += [tf.math.reduce_sum(self.net.losses)]
            losses = tf.convert_to_tensor(losses)
            # Weighted losses
            if loss_weights is not None:
                losses *= loss_weights
            return outputs_, losses

        @tf.function(jit_compile=config.xla_jit)
        def outputs_losses_train(inputs, targets, auxiliary_vars):
            return outputs_losses(
                True, inputs, targets, auxiliary_vars, self.data.losses_train
            )

        @tf.function(jit_compile=config.xla_jit)
        def outputs_losses_test(inputs, targets, auxiliary_vars):
            return outputs_losses(
                False, inputs, targets, auxiliary_vars, self.data.losses_test
            )

        opt = optimizers.get(self.opt_name, learning_rate=lr, decay=decay)

        @tf.function(jit_compile=config.xla_jit)
        def train_step(inputs, targets, auxiliary_vars):
            # inputs and targets are np.ndarray and automatically converted to Tensor.
            with tf.GradientTape() as tape:
                losses = outputs_losses_train(inputs, targets, auxiliary_vars)[1]
                total_loss = tf.math.reduce_sum(losses)
            trainable_variables = (
                self.net.trainable_variables + self.external_trainable_variables
            )
            grads = tape.gradient(total_loss, trainable_variables)
            grads_no_nan = [tf.where(tf.math.is_finite(grad), grad, 10000000) for grad in grads]
            clipped_grads = [tf.clip_by_norm(grad,10) for grad in grads_no_nan]
            #tf.print(clipped_grads, summarize = -1)
            #opt.apply_gradients(zip(grads, trainable_variables))
            opt.apply_gradients(zip(clipped_grads, trainable_variables))

        def train_step_tfp(
            inputs, targets, auxiliary_vars, previous_optimizer_results=None
        ):
            def build_loss():
                losses = outputs_losses_train(inputs, targets, auxiliary_vars)[1]
                return tf.math.reduce_sum(losses)

            trainable_variables = (
                self.net.trainable_variables + self.external_trainable_variables
            )
            return opt(trainable_variables, build_loss, previous_optimizer_results)

        # Callables
        self.outputs = outputs
        self.outputs_losses_train = outputs_losses_train
        self.outputs_losses_test = outputs_losses_test
        self.train_step = (
            train_step
            if not optimizers.is_external_optimizer(self.opt_name)
            else train_step_tfp
        )

Model._compile_tensorflow = custom_compile_tensorflow


# Add initializer for the bias values. Otherwise their initial
# values are zeros.
import deepxde
import tensorflow as tf
from deepxde.nn import regularizers, activations, initializers
from deepxde.nn.tensorflow import NN

class FNN_custom(NN):
    """Fully-connected neural network."""

    def __init__(
        self,
        layer_sizes,
        activation,
        kernel_initializer,
        regularization=None,
        dropout_rate=0,
    ):
        super().__init__()
        self.regularizer = regularizers.get(regularization)
        self.dropout_rate = dropout_rate

        self.denses = []
        if isinstance(activation, list):
            if not (len(layer_sizes) - 1) == len(activation):
                raise ValueError(
                    "Total number of activation functions do not match with sum of hidden layers and output layer!"
                )
            activation = list(map(activations.get, activation))
        else:
            activation = activations.get(activation)
        initializer = initializers.get(kernel_initializer)
        for j, units in enumerate(layer_sizes[1:-1]):
            self.denses.append(
                tf.keras.layers.Dense(
                    units,
                    activation=(
                        activation[j]
                        if isinstance(activation, list)
                        else activation
                    ),
                    kernel_initializer=initializer,
                    bias_initializer=initializer,
                    kernel_regularizer=self.regularizer,
                )
            )
            if self.dropout_rate > 0:
                self.denses.append(tf.keras.layers.Dropout(rate=self.dropout_rate))

        self.denses.append(
            tf.keras.layers.Dense(
                layer_sizes[-1],
                kernel_initializer=initializer,
                bias_initializer=initializer,
                kernel_regularizer=self.regularizer,
            )
        )

    def call(self, inputs, training=False):
        y = inputs
        if self._input_transform is not None:
            y = self._input_transform(y)
        for f in self.denses:
            y = f(y, training=training)
        if self._output_transform is not None:
            y = self._output_transform(inputs, y)
        return y


deepxde.nn.tensorflow.fnn.FNN = FNN_custom

#########################################################################
os.chdir('inverse_problem')

# Import the parameters relevant to the scaling of the data
with open('train_data/parameters.csv', 'r') as csvfile:
    my_reader = csv.reader(csvfile)
    header = next(my_reader)  # Skip the header row
    data_row = next(my_reader)  # Read the data row

# Assign values to variables
t_bar, t_sd,x_mean_bar, x_mean_sd, y_mean_bar, y_mean_sd, \
z_mean_bar, z_mean_sd, x_var_bar, x_var_sd, y_var_bar, y_var_sd, \
z_var_bar, z_var_sd, t_max, t_min, x_mean_max, \
x_mean_min, y_mean_max, y_mean_min, z_mean_max, z_mean_min, \
x_var_max,  x_var_min, y_var_max, y_var_min, \
z_var_max, z_var_min, x_max, x_min, y_max, y_min,\
z_max, z_min  = map(float, data_row)


# Create time domain
geom = dde.geometry.TimeDomain(0.0, 5.0)

def PK_system(t, nn_output):
    # x refers to the central compartment
    # y refers to the peripheral compartment
    x_mean = nn_output[:, 0:1]*(x_mean_max-x_mean_min)+x_mean_min
    x_var =  nn_output[:, 1:2]*(x_var_max-x_var_min)+x_var_min
    y_mean = nn_output[:, 2:3]*(y_mean_max-y_mean_min)+y_mean_min
    y_var =  nn_output[:, 3:4]*(y_var_max-y_var_min)+y_var_min
    z_mean = nn_output[:, 4:5]*(z_mean_max-z_mean_min)+z_mean_min
    z_var = nn_output[:, 5:6]*(z_var_max-z_var_min)+z_var_min
    
    dx_t = dde.grad.jacobian(nn_output, t, i=0)
    dx_var = dde.grad.jacobian(nn_output, t, i=1)
    dy_t = dde.grad.jacobian(nn_output, t, i=2)
    dy_var = dde.grad.jacobian(nn_output, t, i=3)
    dz_t = dde.grad.jacobian(nn_output, t, i=4)
    dz_var = dde.grad.jacobian(nn_output, t, i=5)

    # Sample N values from x,y
    N = 125
    Seed = 1200
    eps = 0.000000001
    
    x_mean += eps
    y_mean += eps
    z_mean += eps

    x_var = tf.math.abs(x_var)
    y_var = tf.math.abs(y_var)
    z_var = tf.math.abs(z_var)
    
    N_par=6
    
    mu_x = tf.math.log((x_mean**2) / tf.math.sqrt((x_mean**2) + (x_var)))
    sigma_x = tf.math.sqrt(tf.math.log(1 + ((x_var) / (x_mean**2))))
    mu_y = tf.math.log((y_mean**2) / tf.math.sqrt((y_mean**2) + (y_var)))
    sigma_y = tf.math.sqrt(tf.math.log(1 + ((y_var) / (y_mean**2))))
    mu_z = tf.math.log((z_mean**2) / tf.math.sqrt((z_mean**2) + (z_var)))
    sigma_z = tf.math.sqrt(tf.math.log(1 + ((z_var) / (z_mean**2))))
    
    # Define a lambda function that takes a tf.Variable() and returns a tf.constant()
    tanh = lambda z:  ((tf.math.exp(1.0) **z - tf.math.exp(1.0) **(-z)) / (tf.math.exp(1.0) **z + tf.math.exp(1.0) **(-z)) )

    combo1 = tf.stack([ 1., tanh(c_x_y), tanh(c_x_z), tanh(c_x_k12), tanh(c_x_k21), tanh(c_x_ke)])
    combo1_reshaped = tf.reshape(combo1, (6, 1))
    combo2 = tf.stack([ tanh(c_x_y), 1., tanh(c_y_z), tanh(c_y_k12), tanh(c_y_k21), tanh(c_y_ke)])
    combo2_reshaped = tf.reshape(combo2, (6, 1))
    combo3 = tf.stack([ tanh(c_x_z), tanh(c_y_z), 1., tanh(c_z_k12), tanh(c_z_k21), tanh(c_z_ke)])
    combo3_reshaped = tf.reshape(combo3, (6, 1))
    combo4 = tf.stack([ tanh(c_x_k12), tanh(c_y_k12), tanh(c_z_k12), 1., tanh(c_k12_k21) , tanh(c_k12_ke)])
    combo4_reshaped = tf.reshape(combo4, (6, 1))
    combo5 = tf.stack([ tanh(c_x_k21), tanh(c_y_k21), tanh(c_z_k21), tanh(c_k12_k21) , 1., tanh(c_k21_ke)])
    combo5_reshaped = tf.reshape(combo5, (6, 1))
    combo6 = tf.stack([ tanh(c_x_ke) , tanh(c_y_ke) , tanh(c_z_ke) , tanh(c_k12_ke) , tanh(c_k21_ke), 1.])
    combo6_reshaped = tf.reshape(combo6, (6, 1))
    
    corr_matrix = tf.transpose(tf.concat([combo1_reshaped, combo2_reshaped,combo3_reshaped, combo4_reshaped,combo5_reshaped, combo6_reshaped],1 ))
    size = tf.shape(mu_x)[0]

    mu = tf.concat([mu_x, 
                   mu_y,
                   mu_z, 
                   tf.fill(dims=[size,1], value=k12),
                   tf.fill(dims=[size,1], value=k21),
                   tf.fill(dims=[size,1], value=ke)], 
                   axis = 1)
    
    S = tf.linalg.diag(tf.concat([sigma_x, 
                                  sigma_y,
                                  sigma_z,
                                  tf.math.abs(tf.fill(dims=[size,1], value = k12_sd+eps)),
                                  tf.math.abs(tf.fill(dims=[size,1], value = k21_sd+eps)),
                                  tf.math.abs(tf.fill(dims=[size,1], value = ke_sd+eps))],
                                  axis = 1))
    cov_mat = tf.linalg.matmul(tf.linalg.matmul(S,corr_matrix),S)

    # Add a small number to the diagonal to make the covariance matrix positive definite
    #cov_mat += 5e-4 * tf.eye(N_par)

    tf.random.set_seed(seed=Seed)
    mvn = tfp.distributions.MultivariateNormalTriL(loc =mu, \
            scale_tril = tfp.experimental.linalg.simple_robustified_cholesky(cov_mat)).sample(N)

    x_samples = tf.transpose(tf.exp(mvn[:,:,0]))
    y_samples = tf.transpose(tf.exp(mvn[:,:,1]))  
    z_samples = tf.transpose(tf.exp(mvn[:,:,2]))
    k12_samples =  tf.transpose(tf.exp(mvn[:,:,3]))
    k21_samples =  tf.transpose(tf.exp(mvn[:,:,4]))
    ke_samples =  tf.transpose(tf.exp(mvn[:,:,5]))
    
    x_bar =  tf.math.reduce_mean(x_samples, axis=1, keepdims=True)
    y_bar =  tf.math.reduce_mean(y_samples, axis=1, keepdims=True)
    z_bar =  tf.math.reduce_mean(z_samples, axis=1, keepdims=True)
    
    dx_rs_unscaled = k21_samples * y_samples - k12_samples * x_samples - ke_samples * x_samples
    dy_rs_unscaled = k12_samples * x_samples - k21_samples * y_samples
    dz_rs_unscaled = ke_samples * x_samples

    dx_rs_mean_unscaled = tf.math.reduce_mean(dx_rs_unscaled, axis=1, keepdims=True)
    dx_rs_mean_scaled = dx_rs_mean_unscaled / (x_mean_max - x_mean_min)

    dy_rs_mean_unscaled = tf.math.reduce_mean(dy_rs_unscaled, axis=1, keepdims=True)
    dy_rs_mean_scaled = dy_rs_mean_unscaled/(y_mean_max - y_mean_min)

    dz_rs_mean_unscaled = tf.math.reduce_mean(dz_rs_unscaled, axis=1, keepdims=True)
    dz_rs_mean_scaled = dz_rs_mean_unscaled/(z_mean_max - z_mean_min)
    
    dx_rs_var_unscaled =  (2/(N-1)) *tf.math.reduce_sum((x_samples - x_bar) *\
                                 (dx_rs_unscaled  - dx_rs_mean_unscaled), axis=1, keepdims=True)
    dx_rs_var_scaled =  dx_rs_var_unscaled / (x_var_max-x_var_min) 
    
    dy_rs_var_unscaled =  (2/(N-1))  * tf.math.reduce_sum((y_samples - y_bar) *\
                                    (dy_rs_unscaled - dy_rs_mean_unscaled), axis=1, keepdims=True)
    dy_rs_var_scaled =  dy_rs_var_unscaled / (y_var_max-y_var_min) 

    dz_rs_var_unscaled = (2/(N-1)) * tf.math.reduce_sum((z_samples - z_bar) *\
                                    (dz_rs_unscaled - dz_rs_mean_unscaled), axis=1, keepdims=True)
    dz_rs_var_scaled =  dz_rs_var_unscaled / (z_var_max-z_var_min) 

    # combined_tensors = tf.concat([t, dx_var, dx_rs_var], axis=-1)
    # combined_tensors = tf.concat([t, dx_t, dx_rs_mean, dx_var, dx_rs_var], axis=-1)
    # tf.print(combined_tensors, summarize =-1)

    loss1 = (dx_t - dx_rs_mean_scaled)
    loss2 = (dx_var - dx_rs_var_scaled)
    loss3 = (dy_t - dy_rs_mean_scaled)
    loss4 = (dy_var - dy_rs_var_scaled)
    loss5 = (dz_t - dz_rs_mean_scaled)
    loss6 = (dz_var - dz_rs_var_scaled)

    return ([loss1,loss2,loss3,loss4,loss5,loss6])
    
# Next, we consider the initial conditions. We need to implement a function, 
# which should return True for points inside the subdomain and False for the points outside.

# def boundary(_, on_initial):
#     return on_initial 

def boundary_initial(x, on_initial):
    return on_initial and np.isclose(x[0], 0.0)
dose = 1.0
ic1 = dde.IC(geom, lambda X: dose, boundary_initial, component=0)
ic2 = dde.IC(geom, lambda X: 0, boundary_initial, component=1)
ic3 = dde.IC(geom, lambda X: 0, boundary_initial, component=2)
ic4 = dde.IC(geom, lambda X: 0, boundary_initial, component=3)
ic5 = dde.IC(geom, lambda X: 0, boundary_initial, component=4)
ic6 = dde.IC(geom, lambda X: 0, boundary_initial, component=5)

# Load train data
def gen_traindata(file_name):
    data = pd.read_csv(file_name)
    return np.vstack(data["t"]) , np.vstack(data["X"]), np.vstack(data["X_var"]), np.vstack(data["Y"]), np.vstack(data["Y_var"]), np.vstack(data["Z"]), np.vstack(data["Z_var"])

# Organize and assign the train data
t_obs, x_obs, x_var_obs, y_obs, y_var_obs, z_obs, z_var_obs = gen_traindata('train_data/PK_simulation.csv')
observe_x = dde.icbc.PointSetBC(t_obs, x_obs, component=0)
observe_x_var = dde.icbc.PointSetBC(t_obs, x_var_obs, component=1)
observe_y = dde.icbc.PointSetBC(t_obs, y_obs, component=2)
observe_y_var = dde.icbc.PointSetBC(t_obs, y_var_obs, component=3)
observe_z = dde.icbc.PointSetBC(t_obs, z_obs, component=4)
observe_z_var = dde.icbc.PointSetBC(t_obs, z_var_obs, component=5)

data = dde.data.PDE(
    geometry = geom,
    pde = PK_system,
    bcs = [ic1, ic2, ic3, ic4, ic5, ic6, observe_x, observe_x_var, observe_y, observe_y_var, 
           observe_z, observe_z_var],
    num_domain=200,
    num_boundary=2)

# Initialize the kinetic parameters
k12 = dde.Variable(0.0)
k12_sd = dde.Variable(0.0)
k21 = dde.Variable(0.0)
k21_sd = dde.Variable(0.0)
ke = dde.Variable(0.0)
ke_sd = dde.Variable(0.0)

c_x_y = dde.Variable(0.0)
c_x_z = dde.Variable(0.0)
c_x_k12 = dde.Variable(0.0)
c_x_k21 = dde.Variable(0.0)
c_x_ke = dde.Variable(0.0)

c_y_z = dde.Variable(0.0)
c_y_k12 = dde.Variable(0.0)
c_y_k21 = dde.Variable(0.0)
c_y_ke = dde.Variable(0.0)

c_z_k12 = dde.Variable(0.0)
c_z_k21 = dde.Variable(0.0)
c_z_ke = dde.Variable(0.0)

c_k12_k21 = dde.Variable(0.0)
c_k12_ke = dde.Variable(0.0)
c_k21_ke = dde.Variable(0.0)

external_trainable_variables=[k12, k12_sd, k21, k21_sd, ke, ke_sd, 
                              c_x_y, c_x_z, c_x_k12, c_x_k21, c_x_ke,
                              c_y_z, c_y_k12, c_y_k21, c_y_ke,
                              c_z_k12, c_z_k21, c_z_ke,
                              c_k12_k21, c_k12_ke, c_k21_ke]

# Callback to report the kinetic parameters 
variable = dde.callbacks.VariableValue(
    external_trainable_variables, period=100, filename="variables.dat", precision=5
)

# Compile the model with the optimizer following an exponential decay approach in the learning rate
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.95,
    staircase=True)

#initializer = tf.keras.initializers.he_normal(seed=100) #"Glorot uniform"
initializer = tf.keras.initializers.GlorotNormal(seed=9827)#
#initializer = tf.keras.initializers.RandomNormal(mean=0.01, stddev=0.005, seed=100)
activation = 'sigmoid' #tf.keras.layers.LeakyReLU(alpha=0.5) #"relu"
regularization = ["l2", 0.01]
net = dde.nn.tensorflow.fnn.FNN([1] + [5] * 2 + [6] , activation, initializer)

loss_weights = [1,1,1,1,1,1,  1,1,1,1,1,1, 20,20, 20,20,20,20] #,20,20] 
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule) # or any other optimizer
model = dde.Model(data, net)
model.compile(optimizer=optimizer, external_trainable_variables=external_trainable_variables, loss_weights=loss_weights) 
losshistory, train_state = model.train(iterations=30000, callbacks=[variable])

t_points = np.reshape(np.linspace(0, 5, 1000), (1000,1))
# Plot predictions of the NN
nn_preds = model.predict(t_points)
preds_df = pd.DataFrame(np.column_stack((t_points,nn_preds)), columns=['time','x_mean', 'x_var', 'y_mean', 'y_var', 'z_mean', 'z_var'])
preds_df.to_csv('results/predictions_2states.csv', index=False)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot()
ax.plot(t_points, nn_preds[:,0], 'b-', label = 'x_mean')
ax.plot(t_obs, x_obs, 'bo', label = 'x_mean_data')
ax.plot(t_points, nn_preds[:,1], 'b--', label = 'x_var')
ax.plot(t_obs, x_var_obs, 'b+', label = 'x_var_data')

ax.plot(t_points, nn_preds[:,2], 'y-', label = 'y_mean')
ax.plot(t_obs, y_obs, 'yo', label = 'y_mean_data')
ax.plot(t_points, nn_preds[:,3], 'y--', label = 'y_var')
ax.plot(t_obs, y_var_obs, 'y+', label = 'y_var_data')

ax.plot(t_points, nn_preds[:,4], 'r-', label = 'z_mean')
ax.plot(t_obs, z_obs, 'ro', label = 'z_mean_data')
ax.plot(t_points, nn_preds[:,5], 'r--', label = 'z_var')
ax.plot(t_obs, z_var_obs, 'r+', label = 'z_var_data')
ax.set_title('Pharmacokinetic System Simulation')
ax.legend()
plt.show()


# model.state_dict()
# pd.DataFrame(np.column_stack((t_obs, model.predict(t_obs))))
# model.opt_name.lr
# # Plot predictions of the NN
# nn_preds2 = model.predict(t_obs)
# # Save the noisy data in a csv file
# df_results = pd.DataFrame({'t': t_obs[:,0], 'X': nn_preds2[:,0], 'X_var':nn_preds2[:,1],
#                    'Y': nn_preds2[:,2], 'Y_var': nn_preds2[:,3]})
# df_results.to_csv('results.csv', index=False)




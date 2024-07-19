
k12=-6.30559e-01; k12_sd=-1.96845e+00; k21= -1.54811e+00;k21_sd=-2.70369e+00; ke=-1.22436e+00;  ke_sd=  -1.46610e+00;
c_x_y=  7.77710e-01; c_x_z= -4.01966e-01;c_x_k12= -2.34619e-02;c_x_k21= -6.65943e-01;c_x_ke=-7.00101e-01; 
c_y_z=   -1.41998e+00;c_y_k12=  9.49603e-01;  c_y_k21= -1.36520e+00;c_y_ke=-2.53149e+00;
c_z_k12= -1.57577e+00;c_z_k21=   6.29742e-01;c_z_ke= 1.84795e+00; 
c_k12_k21=-5.42868e-01;  c_k12_ke=-1.12366e+00; c_k21_ke= 1.06114e+00

mu_x = tf.math.log((k12**2) / tf.math.sqrt((k12**2) + (k12_sd**2)))
sigma_x = tf.math.sqrt(tf.math.log(1 + ((k12_sd**2) / (k12**2))))
mu_y = tf.math.log((k21**2) / tf.math.sqrt((k21**2) + (k21_sd**2)))
sigma_y = tf.math.sqrt(tf.math.log(1 + ((k21_sd**2) / (k21**2))))
mu_z = tf.math.log((ke**2) / tf.math.sqrt((ke**2) + (ke_sd**2)))
sigma_z = tf.math.sqrt(tf.math.log(1 + ((ke_sd**2) / (ke**2))))

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
#corr_matrix2 = tf.transpose(tf.concat([combo1_reshaped, combo1_reshaped,combo1_reshaped, combo1_reshaped,combo1_reshaped, combo1_reshaped],1 ))
size = 1

mu = tf.stack([mu_x, 
   mu_y,
   mu_z, 
   tf.constant(k12),tf.constant(k21), tf.constant(ke)], axis=0)

S = tf.linalg.diag(tf.stack([sigma_x, 
  sigma_y,
  sigma_z,
  tf.math.exp(k12_sd),
  tf.math.exp(k21_sd),
  tf.math.exp(ke_sd)]))
cov_mat = tf.linalg.matmul(tf.linalg.matmul(S,corr_matrix),S)

scale_tri_exp = tfp.experimental.linalg.simple_robustified_cholesky(cov_mat)
scale_tril  = tf.linalg.cholesky(cov_mat)

m1 = np.array([[ 1.5409324e+00,  0.0000000e+00,  0.0000000e+00,  0.0000000e+00,
         0.0000000e+00,  0.0000000e+00],
       [ 7.7038723e-01,  8.9735073e-01,  0.0000000e+00,  0.0000000e+00,
         0.0000000e+00,  0.0000000e+00],
       [-3.5992405e-01, -7.9677391e-01,  3.5367694e-01,  0.0000000e+00,
         0.0000000e+00,  0.0000000e+00],
       [-3.2764017e-03,  1.3896272e-01, -3.2166529e-02,  1.0000000e-03,
         0.0000000e+00,  0.0000000e+00],
       [-3.8989957e-02, -4.3973025e-02, -3.9132934e-02, -1.1007077e-04,
         1.0000000e-03,  0.0000000e+00],
       [-1.3951737e-01, -1.8061635e-01,  3.6820527e-02,  2.9578264e-04,
        -3.6788813e-04,  1.0000000e-03]])
m2 = np.transpose(m1)
cov2 = np.matmul(m1, m2)
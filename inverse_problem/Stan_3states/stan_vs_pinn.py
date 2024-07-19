import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# Define custom legend elements
from matplotlib.lines import Line2D

os.chdir('inverse_problem')

#  get estimated parameters per iteration from PINN
def get_param_estimates(file_name, sheet_name):
     # Read data from xlsx file
    data = pd.read_excel(file_name, sheet_name, index_col=0, header=0)

    # Create an iterable of arrays from the data
    true_params = [data.iloc[1, 0], data.iloc[1, 1], data.iloc[1, 2], data.iloc[1, 3],
                  data.iloc[1, 4], data.iloc[1, 5], data.iloc[1, 6], data.iloc[1, 7],
                  data.iloc[1, 8]]
    
     # Concatenate the arrays in the iterable using vstack()
    stacked_true_params = np.vstack(true_params)
    return stacked_true_params

def get_true_params(file_name, sheet_name):
    # Read data from xlsx file
    data = pd.read_excel(file_name, sheet_name, index_col=0, header=0)
    
    # Create an iterable of arrays from the data
    true_params = [data.iloc[0, 0], data.iloc[0, 1], data.iloc[0, 2], data.iloc[0, 3],
                  data.iloc[0, 4], data.iloc[0, 5], data.iloc[0, 6], data.iloc[0, 7],
                  data.iloc[0, 8]]

    # Concatenate the arrays in the iterable using vstack()
    stacked_true_params = np.vstack(true_params)
    return stacked_true_params

def get_mean(log_mean, log_sd):
    return np.exp(log_mean+(log_sd**2)/2)
def get_sd(log_mean, log_sd):
    return np.sqrt((np.exp(log_sd**2 - 1))*np.exp(2*log_mean+log_sd**2))

log_k12_pinn,log_k12_sd_pinn,log_k21_pinn,log_k21_sd_pinn,log_ke_pinn,log_ke_sd_pinn,c_k12_k21_pinn,c_k12_ke_pinn,c_k21_ke_pinn =\
                                                              get_param_estimates("Stan_3states/pinn_estimates_reverse_3states.xlsx","Estimates")
log_k12_stan,log_k12_sd_stan,log_k21_stan,log_k21_sd_stan,log_ke_stan,log_ke_sd_stan,c_k12_k21_stan,c_k12_ke_stan,c_k21_ke_stan = \
                                                              get_param_estimates("Stan_3states/stan_estimates_reverse_3states.xlsx","Estimates")
log_k12,log_k12_sd,log_k21,log_k21_sd,log_ke,log_ke_sd,c_k12_k21,c_k12_ke,c_k21_ke =  get_true_params("Stan_3states/stan_estimates_reverse_3states.xlsx","Estimates")


k12_pinn = get_mean(log_k12_pinn ,log_k12_sd_pinn)
k12_sd_pinn = get_sd(log_k12_pinn ,log_k12_sd_pinn)
k21_pinn = get_mean(log_k21_pinn,log_k21_sd_pinn)
k21_sd_pinn = get_sd(log_k21_pinn,log_k21_sd_pinn)
ke_pinn =get_mean(log_ke_pinn,log_ke_sd_pinn)
ke_sd_pinn = get_sd(log_ke_pinn,log_ke_sd_pinn)

k12_stan = get_mean(log_k12_stan ,log_k12_sd_stan)
k12_sd_stan = get_sd(log_k12_stan ,log_k12_sd_stan)
k21_stan = get_mean(log_k21_stan,log_k21_sd_stan)
k21_sd_stan = get_sd(log_k21_stan,log_k21_sd_stan)
ke_stan =get_mean(log_ke_stan,log_ke_sd_stan)
ke_sd_stan = get_sd(log_ke_stan,log_ke_sd_stan)

k12 = get_mean(log_k12 ,log_k12_sd)
k12_sd = get_sd(log_k12 ,log_k12_sd)
k21 = get_mean(log_k21,log_k21_sd)
k21_sd = get_sd(log_k21,log_k21_sd)
ke =get_mean(log_ke,log_ke_sd)
ke_sd = get_sd(log_ke,log_ke_sd)

true_values = [k12,k12_sd,k21,k21_sd,ke,ke_sd,c_k12_k21,c_k12_ke,c_k21_ke]
pinn_predictions = [k12_pinn,k12_sd_pinn,k21_pinn,k21_sd_pinn,ke_pinn,ke_sd_pinn,c_k12_k21_pinn,c_k12_ke_pinn,c_k21_ke_pinn]
stan_predictions = [k12_stan,k12_sd_stan,k21_stan,k21_sd_stan,ke_stan,ke_sd_stan,c_k12_k21_stan,c_k12_ke_stan,c_k21_ke_stan]

# Separate parameters into classes (mean, sd, correlation)
mean_true = true_values[:3]
sd_true = true_values[3:6]
correlation_true = true_values[6:]

mean_pinn = pinn_predictions[:3]
sd_pinn = pinn_predictions[3:6]
correlation_pinn = pinn_predictions[6:]

mean_stan = stan_predictions[:3]
sd_stan = stan_predictions[3:6]
correlation_stan = stan_predictions[6:]

# Separate parameters into classes (mean, sd, correlation)
mean_true = true_values[:3]
sd_true = true_values[3:6]
correlation_true = true_values[6:]

mean_pinn = pinn_predictions[:3]
sd_pinn = pinn_predictions[3:6]
correlation_pinn = pinn_predictions[6:]

mean_stan = stan_predictions[:3]
sd_stan = stan_predictions[3:6]
correlation_stan = stan_predictions[6:]


# Create the plot
fig, ax = plt.subplots(figsize=(9,6))
plt.style.use('tableau-colorblind10')


linetypes_markers_legend = [Line2D([0], [0], color='#FF800E',  marker='o', alpha=0.5,label='D-PINN-mean'),
                            Line2D([0], [0], color='#006BA4',  marker='o',alpha=0.5, label='Stan-mean'),
                            Line2D([0], [0], color='#FF800E',  marker='s', alpha=0.5,label='D-PINN-standard deviation'),
                            Line2D([0], [0], color='#006BA4',  marker='s', alpha=0.5,label='Stan-standard deviation'),
                            Line2D([0], [0], color='#FF800E',  marker='^',alpha=0.5, label='D-PINN-correlation'),
                            Line2D([0], [0], color='#006BA4',  marker='^', alpha=0.5,label='Stan-correlation'),] 


# Plotting means
ax.scatter(mean_true, mean_pinn, marker='o', s=150, color='#FF800E', alpha=0.5, label='D-PINN-mean')
ax.scatter(mean_true, mean_stan, marker='o', s=150, color='#006BA4', alpha=0.5, label='Stan-mean')

# Plotting sds
ax.scatter(sd_true, sd_pinn, marker='s', s=150, color='#FF800E', alpha=0.5, label='D-PINN-standard deviation')
ax.scatter(sd_true, sd_stan, marker='s', s=150, color='#006BA4', alpha=0.5, label='Stan-standard deviation')

# Plotting correlations
ax.scatter(correlation_true, correlation_pinn, marker='^', s=150, color='#FF800E', alpha=0.5, label='D-PINN-correlation')
ax.scatter(correlation_true, correlation_stan, marker='^', s=150, color='#006BA4', alpha=0.5, label='Stan-correlation')

# Plotting the identity line
ax.plot([-1, 1], [-1, 1], '--k', label='Identity Line (y=x)')

# Set labels and title
ax.set_xlabel('True Values', fontsize=14)
ax.set_ylabel('Predicted Values', fontsize=14)

# Set the axis limits
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0 * 0.9, box.y0 *2.3, box.width * 1.1, box.height * 0.9])

# Add both legends to the plot
legend = ax.legend(handles=linetypes_markers_legend, bbox_to_anchor=(0.5, -0.15), loc="upper center",
                    borderaxespad=0.0, ncol=3, fancybox=True, shadow=True, fontsize=12)

# Add titles to each legend
legend.set_title('Estimation Scheme and Variable Type',prop={'size':'large'})

# Save the figure with high resolution and include the legend box
plt.savefig('plots/Stan_vs_PINN_3states.png', format='png', dpi=600)

# Show the plot
plt.show()
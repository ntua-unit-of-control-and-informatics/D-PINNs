import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import pandas as pd
import numpy as np
import csv

os.chdir('/Users/vassilis/Documents/GitHub/Lotka_Voltera/PK_PINN/Forward_problem')

# A function to estimate the Confidence intervals
def estimate_CI(data, N_samples = 1000):
    mean = data.iloc[:,0]
    variance = data.iloc[:,1] 
    samples = np.random.multivariate_normal(mean, np.diag(np.sqrt(variance)), N_samples)
    lower_bound = np.percentile(samples,2.5, axis = 0)
    upper_bound = np.percentile(samples,97.5, axis = 0)

    return lower_bound, upper_bound

def moving_average(data, window_size):
    cumsum = np.cumsum(data, dtype=float)
    cumsum[window_size:] = cumsum[window_size:] - cumsum[:-window_size]
    return cumsum[window_size - 1:] / window_size


# Function to create individual plots
def create_plot(ax, time_points, mean_odes, ode_ci_lower, ode_ci_upper, pinn_ci_lower, pinn_ci_upper, mean_pinns, \
                color_odes, color_pinns, y_min, y_max, x_title,  plot_title = np.nan, window_size=10):
    
    # Smooth the upper and lower bounds using moving average
    smooth_ode_ci_lower = moving_average(ode_ci_lower, window_size)
    smooth_ode_ci_upper = moving_average(ode_ci_upper, window_size)
    smooth_pinn_ci_lower = moving_average(pinn_ci_lower, window_size)
    smooth_pinn_ci_upper = moving_average(pinn_ci_upper, window_size)

    ax.plot(time_points, mean_odes, linestyle='dashed', color=color_odes, label='ODEs Mean')
    ax.fill_between(time_points[:len(smooth_ode_ci_lower)], smooth_ode_ci_lower, smooth_ode_ci_upper, color=color_odes, alpha=0.2, label='ODEs 95% Quantiles')
    ax.plot(time_points, mean_pinns, linestyle='dashed', color=color_pinns, label='PINN Mean')
    ax.fill_between(time_points[:len(smooth_pinn_ci_lower)], smooth_pinn_ci_lower, smooth_pinn_ci_upper, color=color_pinns, alpha=0.2, label='PINN 95% Quantiles')
    ax.set_ylim(bottom =y_min, top = y_max )



    if x_title == True:
        ax.set_xlabel('Time')
    
    if isinstance(plot_title, str):
        ax.set_title(plot_title)


def create_3x3_grid(odes_data_list, pinn_data_list, time_points):
    # Set up the 3x3 grid

    fig, axs = plt.subplots(3, 3, figsize=(9, 9), sharex='all', sharey='row')

    for i in range(3):
        if i == 0:
            plot_title = ['Central compartment', 'Peripheral compartment', 'Excreta']
        else:
            plot_title = np.repeat(np.nan, 3)

        if i == 2:
            x_title = True
        else:
            x_title = False

        odes_data = odes_data_list[i]
        pinn_data = pinn_data_list[i]

        x_odes_data = odes_data.iloc[:, 2:4]
        x_odes_ci_lower, x_odes_ci_upper = estimate_CI(x_odes_data)

        x_pinns_data = pinn_data.iloc[:, 2:4]
        x_pinns_ci_lower, x_pinns_ci_upper = estimate_CI(x_pinns_data)

        y_odes_data = odes_data.iloc[:, 4:6]
        y_odes_ci_lower, y_odes_ci_upper = estimate_CI(y_odes_data)

        y_pinns_data = pinn_data.iloc[:, 4:6]
        y_pinns_ci_lower, y_pinns_ci_upper = estimate_CI(y_pinns_data)

        z_odes_data = odes_data.iloc[:, 6:8]
        z_odes_ci_lower, z_odes_ci_upper = estimate_CI(z_odes_data)

        z_pinns_data = pinn_data.iloc[:, 6:8]
        z_pinns_ci_lower, z_pinns_ci_upper = estimate_CI(z_pinns_data)

        y_min = 0
        y_max = 1.2*max(max(x_odes_ci_upper),max(y_odes_ci_upper),max(z_odes_ci_upper), max(x_pinns_ci_upper), max(y_pinns_ci_upper), max(z_pinns_ci_upper))

        create_plot(axs[i,0], time_points, x_odes_data['X'], x_odes_ci_lower, x_odes_ci_upper, x_pinns_ci_lower, x_pinns_ci_upper,
                x_pinns_data['X'], 'blue', 'yellow', y_min, y_max, x_title = x_title,  plot_title = plot_title[0])
        

        create_plot(axs[i,1], time_points, y_odes_data['Y'], y_odes_ci_lower, y_odes_ci_upper, y_pinns_ci_lower, y_pinns_ci_upper,
                    y_pinns_data['Y'], 'blue', 'yellow',  y_min, y_max, x_title = x_title, plot_title = plot_title[1])

 
        create_plot(axs[i,2], time_points, z_odes_data['Z'], z_odes_ci_lower, z_odes_ci_upper, z_pinns_ci_lower, z_pinns_ci_upper,
                    z_pinns_data['Z'], 'blue', 'yellow',  y_min, y_max, x_title = x_title, plot_title = plot_title[2])
        
        # Set x-axis and y-axis labels
        axs[i, 0].set_ylabel(f'Dose = {doses[i]}', fontsize = 14)
        axs[2, i].set_xlabel('Time', fontsize = 14)

    custom_lines = [Line2D([0], [0], color='blue', lw=2, linestyle='dashed'),
            Line2D([0], [0], color='blue', lw=2),
            Line2D([0], [0], color='yellow', lw=2, linestyle='dashed'),
            Line2D([0], [0], color='yellow', lw=2)]

    # Create a legend for the whole figure
    fig.legend(custom_lines, ['ODEs Mean', 'ODEs 95% Quantiles', 'PINN Mean', 'PINN 95% Quantiles'], 
               bbox_to_anchor=(0.5, -0.0), loc='lower center', 
               ncol=4, fancybox=True, shadow=True, fontsize = 14)
    
    # Adjust layout
    #plt.tight_layout(rect=[0, 0.00, 0, 0.00])

    # Show the plot
    plt.show()
        

odes_samples = pd.read_csv('train_data/ODE_samples.csv')
pinn_samples = pd.read_csv('train_data/pinn_samples.csv')

# Import the parameters relevant to the scaling of the data
with open('train_data/parameters_forward.csv', 'r') as csvfile:
    my_reader = csv.reader(csvfile)
    header = next(my_reader)  # Skip the header row
    data_row = next(my_reader)  # Read the data row

# Assign values to variables
dose_max, dose_min, t_max, t_min, x_mean_max, \
x_mean_min, y_mean_max, y_mean_min, z_mean_max, z_mean_min, \
x_var_max,  x_var_min, y_var_max, y_var_min, \
z_var_max, z_var_min = map(float, data_row)

doses =np.array([6.5,27.3, 46.7])
method =  "no_scale"

if method == "log":
    odes_samples['dose']   = np.exp(odes_samples['dose'])
    odes_samples['t']      = np.exp(odes_samples['t'] )
    odes_samples['X']      = np.exp(odes_samples['X'])
    odes_samples['X_var']  = np.exp(odes_samples['X_var'] )
    odes_samples['Y']      = np.exp(odes_samples['Y'] )
    odes_samples['Y_var']  = np.exp(odes_samples['Y_var'])
    odes_samples['Z']      = np.exp(odes_samples['Z'])
    odes_samples['Z_var']  = np.exp(odes_samples['Z_var']) 

    pinn_samples['dose']  = np.exp(pinn_samples['dose'])
    pinn_samples['t']     = np.exp(pinn_samples['t'] )
    pinn_samples['X']     = np.exp(pinn_samples['X'])
    pinn_samples['X_var'] = np.exp(pinn_samples['X_var'] )
    pinn_samples['Y']     = np.exp(pinn_samples['Y'] )
    pinn_samples['Y_var'] = np.exp(pinn_samples['Y_var'] )
    pinn_samples['Z']     = np.exp(pinn_samples['Z'] )
    pinn_samples['Z_var'] = np.exp(pinn_samples['Z_var'] )
elif method == "log_min_max":
    odes_samples['dose']   = np.exp(odes_samples['X'] * (dose_max-dose_min)+dose_min)
    odes_samples['t']      = np.exp(odes_samples['t'] * (t_max-t_min)+t_min)
    odes_samples['X']      = np.exp(odes_samples['X'] * (x_mean_max-x_mean_min)+x_mean_min)
    odes_samples['X_var']  = np.exp(odes_samples['X_var'] * (x_var_max-x_var_min)+x_var_min)
    odes_samples['Y']      = np.exp(odes_samples['Y'] * (y_mean_max-y_mean_min)+y_mean_min)
    odes_samples['Y_var']  = np.exp(odes_samples['Y_var'] * (y_var_max-y_var_min)+y_var_min)
    odes_samples['Z']      = np.exp(odes_samples['Z'] * (z_mean_max-z_mean_min)+z_mean_min)
    odes_samples['Z_var']  = np.exp(odes_samples['Z_var'] * (z_var_max-z_var_min)+z_var_min)

    pinn_samples['X']     = np.exp(pinn_samples['X'] * (x_mean_max-x_mean_min)+x_mean_min)
    pinn_samples['X_var'] = np.exp(pinn_samples['X_var'] * (x_var_max-x_var_min)+x_var_min)
    pinn_samples['Y']     = np.exp(pinn_samples['Y'] * (y_mean_max-y_mean_min)+y_mean_min)
    pinn_samples['Y_var'] = np.exp(pinn_samples['Y_var'] * (y_var_max-y_var_min)+y_var_min)
    pinn_samples['Z']     = np.exp(pinn_samples['Z'] * (z_mean_max-z_mean_min)+z_mean_min)
    pinn_samples['Z_var'] = np.exp(pinn_samples['Z_var'] * (z_var_max-z_var_min)+z_var_min)
elif method == "min_max":    
    odes_samples['dose']  = odes_samples['dose'] *(dose_max-dose_min)+dose_min
    odes_samples['t']     = odes_samples['t'] * (t_max-t_min)+t_min
    odes_samples['X']     = odes_samples['X'] * (x_mean_max-x_mean_min)+x_mean_min
    odes_samples['X_var'] = odes_samples['X_var'] * (x_var_max-x_var_min)+x_var_min
    odes_samples['Y']     = odes_samples['Y'] * (y_mean_max-y_mean_min)+y_mean_min
    odes_samples['Y_var'] = odes_samples['Y_var'] * (y_var_max-y_var_min)+y_var_min
    odes_samples['Z']     = odes_samples['Z'] * (z_mean_max-z_mean_min)+z_mean_min
    odes_samples['Z_var'] = odes_samples['Z_var'] * (z_var_max-z_var_min)+z_var_min
    
    pinn_samples['dose'] = pinn_samples['dose'] * (dose_max-dose_min)+dose_min
    pinn_samples['t'] = pinn_samples['t'] *  (t_max-t_min)+t_min
    pinn_samples['X'] = pinn_samples['X'] * (x_mean_max-x_mean_min)+x_mean_min
    pinn_samples['X_var'] = pinn_samples['X_var'] * (x_var_max-x_var_min)+x_var_min
    pinn_samples['Y'] = pinn_samples['Y'] * (y_mean_max-y_mean_min)+y_mean_min
    pinn_samples['Y_var'] = pinn_samples['Y_var'] * (y_var_max-y_var_min)+y_var_min
    pinn_samples['Z'] = pinn_samples['Z'] * (z_mean_max-z_mean_min)+z_mean_min
    pinn_samples['Z_var'] = pinn_samples['Z_var'] * (z_var_max-z_var_min)+z_var_min
elif method == "min_max_no_dosetime":    
    odes_samples['X']     = odes_samples['X'] * (x_mean_max-x_mean_min)+x_mean_min
    odes_samples['X_var'] = odes_samples['X_var'] * (x_var_max-x_var_min)+x_var_min
    odes_samples['Y']     = odes_samples['Y'] * (y_mean_max-y_mean_min)+y_mean_min
    odes_samples['Y_var'] = odes_samples['Y_var'] * (y_var_max-y_var_min)+y_var_min
    odes_samples['Z']     = odes_samples['Z'] * (z_mean_max-z_mean_min)+z_mean_min
    odes_samples['Z_var'] = odes_samples['Z_var'] * (z_var_max-z_var_min)+z_var_min
    
    pinn_samples['X'] = pinn_samples['X'] * (x_mean_max-x_mean_min)+x_mean_min
    pinn_samples['X_var'] = pinn_samples['X_var'] * (x_var_max-x_var_min)+x_var_min
    pinn_samples['Y'] = pinn_samples['Y'] * (y_mean_max-y_mean_min)+y_mean_min
    pinn_samples['Y_var'] = pinn_samples['Y_var'] * (y_var_max-y_var_min)+y_var_min
    pinn_samples['Z'] = pinn_samples['Z'] * (z_mean_max-z_mean_min)+z_mean_min
    pinn_samples['Z_var'] = pinn_samples['Z_var'] * (z_var_max-z_var_min)+z_var_min    
elif method == "no_scale":
    print("No unscaling is needed")   

odes_low_dose = odes_samples.loc[odes_samples['dose'] == doses[0]]
odes_mid_dose = odes_samples.loc[odes_samples['dose'] == doses[1]]
odes_high_dose = odes_samples.loc[odes_samples['dose'] == doses[2]]

odes_data_list = [odes_low_dose, odes_mid_dose, odes_high_dose]

pinn_low_dose = pinn_samples.loc[np.isclose(pinn_samples['dose'],doses[0])]
pinn_mid_dose = pinn_samples.loc[np.isclose(pinn_samples['dose'], doses[1])]
pinn_high_dose = pinn_samples.loc[np.isclose(pinn_samples['dose'], doses[2])]

pinn_data_list = [pinn_low_dose, pinn_mid_dose, pinn_high_dose]

time_points = odes_low_dose['t']

create_3x3_grid(odes_data_list, pinn_data_list, time_points)




    

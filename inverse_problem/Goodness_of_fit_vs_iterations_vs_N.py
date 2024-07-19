import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

os.chdir('inverse_problem')

#  get estimated parameters per iteration from PINN
def get_PINN_param_estimates(file_name):
    data = pd.read_csv(file_name)

    # Create an iterable of arrays from the data
    true_params = [data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], data.iloc[:, 3],
                  data.iloc[:, 4], data.iloc[:, 5], data.iloc[:, 6], data.iloc[:, 19],
                  data.iloc[:, 20], data.iloc[:, 21]]
    
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

# Implementation of the mean absolute percentage error
def mape( y_pred, y_true): 
    y_pred = np.array([y_pred])
    return 100*(np.abs((y_true - y_pred)/ y_true)) 


def calculate_loss(file_name):
    # Load predicted data for N = 25
    iteration,k12_pred,k12_sd_pred,k21_pred,k21_sd_pred,ke_pred,ke_sd_pred,c_k12_k21_pred,c_k12_ke_pred,c_k21_ke_pred =  get_PINN_param_estimates(file_name)
    k12,k12_sd,k21,k21_sd,ke,ke_sd,c_k12_k21,c_k12_ke,c_k21_ke = get_true_params("results/pinn_estimates_reverse_3states.xlsx", "Estimates")    

    my_array =  np.array([])
    for i in range(len(iteration)):
        #For each iteration we want to calculate the sum of MSE between real and predicted parameters
        iter_mse = mape(k12_pred[i], k12) + mape(np.abs(k12_sd_pred[i]), k12_sd)+\
        mape(k21_pred[i], k21) + mape(np.abs(k21_sd_pred[i]), k21_sd) + \
        mape(ke_pred[i], ke) + mape(np.abs(ke_sd_pred[i]), ke_sd)+\
        mape( np.tanh(c_k12_k21_pred[i]), c_k12_k21) + mape( np.tanh(c_k12_ke_pred[i]), c_k12_ke) +\
        mape( np.tanh(c_k21_ke_pred[i]), c_k21_ke)
        my_array = np.append(my_array,iter_mse/9)
    return my_array

iteration_1000,k12_pred,k12_sd_pred,k21_pred,k21_sd_pred,ke_pred,ke_sd_pred,c_k12_k21_pred,c_k12_ke_pred,c_k21_ke_pred =  get_PINN_param_estimates("csv_files/N=1000.csv")
iteration,k12_pred,k12_sd_pred,k21_pred,k21_sd_pred,ke_pred,ke_sd_pred,c_k12_k21_pred,c_k12_ke_pred,c_k21_ke_pred =  get_PINN_param_estimates("csv_files/N=25.csv")

N_25 = calculate_loss("csv_files/N=25.csv")
N_50 = calculate_loss("csv_files/N=50.csv")
N_75 = calculate_loss("csv_files/N=75.csv")
N_100 = calculate_loss("csv_files/N=100.csv")
N_125 = calculate_loss("csv_files/N=125.csv")
N_150 = calculate_loss("csv_files/N=150.csv")
N_175 = calculate_loss("csv_files/N=175.csv")
N_200 = calculate_loss("csv_files/N=200.csv")
N_300 = calculate_loss("csv_files/N=300.csv")
N_500 = calculate_loss("csv_files/N=500.csv")
N_1000 = calculate_loss("csv_files/N=1000.csv")

# Define custom legend elements
from matplotlib.lines import Line2D
plt.style.use('tableau-colorblind10')
linetypes_markers_legend = [Line2D([0], [0], color= '#006BA4', linestyle='solid', linewidth=2.0, label='N=25'),
                            Line2D([0], [0], color='#FF800E', linestyle='solid',  linewidth=2.0,label='N=100'),
                            Line2D([0], [0], color= '#595959', linestyle='solid', linewidth=2.0, label='N=150'),
                            Line2D([0], [0], color='#C85200', linestyle='solid',  linewidth=2.0,label='N=300')] 

fig, ax = plt.subplots(figsize=(9,6))

ax.plot(iteration, N_25, color= '#006BA4' , linestyle='solid', linewidth=2.0, label='N=25')
ax.plot(iteration, N_100, color='#FF800E', linestyle='solid', linewidth=2.0, label='N=100')
ax.plot(iteration, N_150, color= '#595959', linestyle='solid', linewidth=2.0, label='N=150')
ax.plot(iteration, N_300, color='#C85200', linestyle='solid', linewidth=2.0, label='N=300')

# Add x-axis title
ax.set_xlabel('Iterations', fontsize=14)
ax.set_ylabel('Mean Absolute Percentage Error (%)', fontsize=14)

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0 * 0.9, box.y0 *2.2, box.width * 1.0, box.height * 0.95])

# Add both legends to the plot
legend = ax.legend(handles=linetypes_markers_legend, bbox_to_anchor=(0.5, -0.15), loc="upper center",
                    borderaxespad=0.0, ncol=4, fancybox=True, shadow=True, fontsize=14)

# Add titles to each legend
legend.set_title('Number of Sampled Points',prop={'size':'large'})


# Save the figure with high resolution and include the legend box
plt.savefig('plots/MAPE_vs_N_vs_Iterations.png', format='png', dpi=600)

# Display the plot
plt.show()
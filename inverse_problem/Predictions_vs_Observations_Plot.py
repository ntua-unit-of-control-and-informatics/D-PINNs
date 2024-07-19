import numpy as np
import pandas as pd
import os
import csv
import matplotlib.pyplot as plt

os.chdir('/Users/ptsir/Documents/GitHub/Lotka_Voltera/PK_PINN/Multiple_parameters_estimation/3_states')

# Load train data
def gen_traindata(file_name):
    data = pd.read_csv(file_name)
    return np.vstack(data.iloc[:,0]) , np.vstack(data.iloc[:,1]), np.vstack(data.iloc[:,2]), np.vstack(data.iloc[:,3]), np.vstack(data.iloc[:,4]), np.vstack(data.iloc[:,5]), np.vstack(data.iloc[:,6])

t_obs, x_obs, x_var_obs, y_obs, y_var_obs, z_obs, z_var_obs = gen_traindata('PK_simulation.csv')

# Load predicted data
t_points, x_mean, x_var, y_mean, y_var, z_mean, z_var =  gen_traindata('predictions_3states.csv')

# Define custom legend elements
from matplotlib.lines import Line2D

linetypes_markers_legend = [Line2D([0], [0], color='0', linestyle='-', label='Predicted mean'),
                            Line2D([0], [0], color='0', marker='o', linestyle='None', label='Observed mean'),
                            Line2D([0], [0], color='0', linestyle='--', label='Predicted variance'),
                            Line2D([0], [0], color='0', marker='+', linestyle='None', label='Observed variance')] 



colors_legend = [Line2D([0], [0], color='#006BA4', linestyle='-', label='Central', linewidth=2.0),
                   Line2D([0], [0], color='#FF800E', linestyle='-', label='Peripheral', linewidth=2.0),
                   Line2D([0], [0], color='#595959', linestyle='-', label='Excreta', linewidth=2.0)]


fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot()

ax.plot(t_points, x_mean, color='#006BA4', linestyle='-', label = 'x_mean', linewidth=2.0)
ax.plot(t_obs, x_obs, color='#006BA4', marker='o', linestyle='None', label = 'x_mean_data', markersize=10)
ax.plot(t_points, x_var, color='#006BA4', linestyle='--', label = 'x_var', linewidth=2.0)
ax.plot(t_obs, x_var_obs, color='#006BA4', marker='+', linestyle='None' , label = 'x_var_data', markersize=10)

ax.plot(t_points, y_mean, color='#FF800E', linestyle='-' , label = 'y_mean', linewidth=2.0)
ax.plot(t_obs, y_obs, color='#FF800E', marker='o', linestyle='None', label = 'y_mean_data', markersize=10)
ax.plot(t_points, y_var, color='#FF800E', linestyle='--', label = 'y_var', linewidth=2.0)
ax.plot(t_obs, y_var_obs, color='#FF800E', marker='+', linestyle='None', label = 'y_var_data', markersize=10)

ax.plot(t_points, z_mean, color='#595959', linestyle='-', label = 'z_mean', linewidth=2.0)
ax.plot(t_obs, z_obs, color='#595959', marker='o', linestyle='None', label = 'z_mean_data', markersize=10)
ax.plot(t_points, z_var, color='#595959', linestyle='--', label = 'z_var', linewidth=2.0)
ax.plot(t_obs, z_var_obs, color='#595959', marker='+', linestyle='None', label = 'z_var_data', markersize=10)

# Add x-axis title
ax.set_xlabel('Time', fontsize=14)
ax.set_ylabel('Value of State Variables', fontsize=14)
ax.tick_params(axis='both', labelsize=14)

# Shrink current axis's height by 10% on the bottom
box = ax.get_position()
ax.set_position([box.x0  , box.y0*2.0 ,
                 box.width, box.height*0.95])

# Add both legends to the plot
legend1 = ax.legend(handles=linetypes_markers_legend, bbox_to_anchor=(-0.07, -0.10),#, 0.5, 0.5),
                    loc="upper left",
                    fontsize=18,
                    borderaxespad=0.0, ncol=2, fancybox=True, shadow=True)
legend2 = ax.legend(handles=colors_legend, bbox_to_anchor=(1.07, -0.10),#, -0.5, 0.5),
                    loc="upper right",
                    fontsize=18,
                    borderaxespad=0.0, ncol=3, fancybox=True, shadow=True)

# Add titles to each legend
legend2.set_title('State Variables', prop={'size':'xx-large'})
# Add the legends to the axis explicitly
ax.add_artist(legend1)
ax.add_artist(legend2)
#plt.savefig('Three_states_problem.png', format='png', dpi=600)

plt.show()

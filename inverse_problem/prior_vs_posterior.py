import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import seaborn as sns
import os

os.chdir('inverse_problem')

# Set seed for reproducibility
np.random.seed(42)

# Parameters for the lognormal distributions
pop_params_1 = {'mean':  -0.712757537136586, 'sigma': 0.1980422004353651}
sampled_params_1 = {'mean': -0.662563627, 'sigma': 0.18279864}
predicted_params_1 = {'mean': -6.62E-01, 'sigma': 1.74E-01}

pop_params_2 = {'mean':  -1.629048269010741, 'sigma': 0.1980422004353651}
sampled_params_2 = {'mean': -1.610401606, 'sigma':0.164971772}
predicted_params_2 = {'mean': -1.66E+00, 'sigma':1.57E-01}

pop_params_3 = {'mean': -1.2235831609025767, 'sigma':0.1980422004353651}
sampled_params_3 = {'mean': -1.256299102, 'sigma':0.20740227}
predicted_params_3 = {'mean': -1.23E+00, 'sigma': 2.08E-01}

# Number of samples
num_samples = 1000

# Generate samples from lognormal distributions
pop_samples_1 = np.exp(np.random.normal(loc=pop_params_1['mean'], scale=pop_params_1['sigma'], size=num_samples))
sampled_samples_1 = np.exp(np.random.normal(loc=sampled_params_1['mean'], scale=sampled_params_1['sigma'], size=num_samples))
predicted_samples_1 = np.exp(np.random.normal(loc=predicted_params_1['mean'], scale=predicted_params_1['sigma'], size=num_samples))

pop_samples_2 = np.exp(np.random.normal(loc=pop_params_2['mean'], scale=pop_params_2['sigma'], size=num_samples))
sampled_samples_2 = np.exp(np.random.normal(loc=sampled_params_2['mean'], scale=sampled_params_2['sigma'], size=num_samples))
predicted_samples_2 = np.exp(np.random.normal(loc=predicted_params_2['mean'], scale=predicted_params_2['sigma'], size=num_samples))

pop_samples_3 = np.exp(np.random.normal(loc=pop_params_3['mean'], scale=pop_params_3['sigma'], size=num_samples))
sampled_samples_3 = np.exp(np.random.normal(loc=sampled_params_3['mean'], scale=sampled_params_3['sigma'], size=num_samples))
predicted_samples_3 = np.exp(np.random.normal(loc=predicted_params_3['mean'], scale=predicted_params_3['sigma'], size=num_samples))

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
plt.style.use('tableau-colorblind10')

# Plot density plots for the first parameter set
sns.kdeplot(pop_samples_1, label='Population Distribution', color= '#595959', fill=True, alpha=0.5, ax=axes[0, 0])
sns.kdeplot(sampled_samples_1, label='Sampled Distribution',color='#FF800E', fill=True, alpha=0.5, ax=axes[0, 0])
sns.kdeplot(predicted_samples_1, label='Predicted Distribution',color='#006BA4', fill=True, alpha=0.5, ax=axes[0, 0])
axes[0, 0].set_title(r'$k_{12}$', fontsize=20)

# Plot density plots for the second parameter set
sns.kdeplot(pop_samples_2, label='Population Distribution', color= '#595959', fill=True, alpha=0.5, ax=axes[0, 1])
sns.kdeplot(sampled_samples_2, label='Sampled Distribution',color='#FF800E', fill=True, alpha=0.5, ax=axes[0, 1])
sns.kdeplot(predicted_samples_2, label='Predicted Distribution',color='#006BA4', fill=True, alpha=0.5, ax=axes[0, 1])
axes[0, 1].set_title(r'$k_{21}$', fontsize=20)

# Plot density plots for the third parameter set
sns.kdeplot(pop_samples_3, label='Population Distribution', color= '#595959', fill=True, alpha=0.5, ax=axes[1, 0])
sns.kdeplot(sampled_samples_3, label='Sampled Distribution',color='#FF800E', fill=True, alpha=0.5, ax=axes[1, 0])
sns.kdeplot(predicted_samples_3, label='Predicted Distribution',color='#006BA4', fill=True, alpha=0.5, ax=axes[1, 0])
axes[1, 0].set_title(r'$k_{e}$', fontsize=20)

# Remove the empty subplot in the second row
fig.delaxes(axes[1, 1])

# Add labels
axes[1, 0].set_xlabel('Value (1/h)', fontsize=14)
axes[0, 0].set_ylabel('Density', fontsize=14)
axes[0, 1].set_ylabel('', fontsize=14)
axes[1, 0].set_ylabel('', fontsize=14)

# Legend adjustment
box = axes[1, 0].get_position()
axes[1, 0].set_position([box.x0 * 2.6, box.y0 * 1.5, box.width * 0.9, box.height * 1.0])
axes[0, 0].set_position([box.x0 * 0.9, box.y0 * 5.5, box.width * 0.9, box.height * 1.0])
axes[0, 1].set_position([box.x0 * 4.5, box.y0 * 5.5, box.width * 0.9, box.height * 1.0])

# Add legend in the middle below the plots
axes[1, 0].legend(loc='center', bbox_to_anchor=(0.5, -0.3), fancybox=True, shadow=True, ncol=3, fontsize=14)

# Save the figure with high resolution and include the legend box
plt.savefig('plots/prior_posterior.png', format='png', dpi=600)

# Show the plot
plt.show()

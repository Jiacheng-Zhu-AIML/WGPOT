"""
WGPOT
Wasserstein Distance and Optimal Transport Map
of Gaussian Processes

Jiacheng Zhu
jzhu4@andrew.cmu.edu
"""
import numpy as np
import scipy.io
import scipy.linalg
import pickle
from matplotlib import pyplot as plt
from wgpot import GP_W_barycenter, Wasserstein_GP, logmap, expmap
from utils import Plot_GP

# Notice: Load dataset
#   Load all the GP data
file_name = 'data/GP_data.pkl'
file_open = open(file_name, 'rb')
gp_list = pickle.load(file_open)
file_open.close()

x_file_name = 'data/index_days.pkl'
x_file_open = open(x_file_name, 'rb')
x_days = pickle.load(x_file_open)
x_file_open.close()

# Notice: Visualize all the GPs
#  Plot all the GPs
fig = plt.figure(1)
mean_alpha = 0.1
var_alpha = 0.02
for index, gp in enumerate(gp_list):
    mu, K = gp
    if index == 0:
        Plot_GP(plt, x_days.T, mu, K, 'b', mean_alpha, var_alpha, 'GPs')
    else:
        Plot_GP(plt, x_days.T, mu, K, 'b', mean_alpha, var_alpha)
    # break
plt.xlabel('days')
plt.ylabel('Temperature')


# Notice: Compute the Wasserstein distance of two GPs
gp_0 = gp_list[0]
gp_1 = gp_list[1]
wd_gp = Wasserstein_GP(gp_0, gp_1)
print('The Wasserstein distance of two GPs is ', wd_gp)

# Notice: Compute the Wasserstein Barycenter of this set of GPs
mu_bc, K_bc = GP_W_barycenter(gp_list)
Plot_GP(plt, x_days.T, mu_bc, K_bc, 'r', 1, 0.5, 'Barycenter')
plt.legend()
plt.title('The populations of GPs in blue. The Wasserstein barycenter in red')
plt.savefig('data/barycenter_result.png', bbox_inches='tight')

# Notice: Obtain the optimal transport map between two GPs
gp_0_mu, gp_0_K = gp_list[4]
gp_1_mu, gp_1_K = gp_list[59]
gp_0_mu = - gp_0_mu     # Manipulate the data to get interesting results

# Notice: Plot the two GPs
fig = plt.figure(2)
Plot_GP(plt, x_days.T, gp_0_mu, gp_0_K, 'r', 1, 0.5, 'GP_0')
Plot_GP(plt, x_days.T, gp_1_mu, gp_1_K, 'b', 1, 0.5, 'GP_1')

# Notice: Obtain the push forward of GPs
#   It's the elements on the principal geodesic
v_mu, v_T = logmap(gp_0_mu, gp_0_K, gp_1_mu, gp_1_K)
for t in [0.2, 0.4, 0.6, 0.8, 1.0]:
    v_mu_t = t * v_mu
    v_T_t = t * v_T
    q_mu, q_K = expmap(gp_1_mu, gp_1_K, v_mu_t, v_T_t)
    Plot_GP(plt, x_days.T, q_mu, q_K, 'orange', 0.5, 0.25, 'geodesic t=' + str(t))
plt.xlabel('days')
plt.ylabel('Temperature')
plt.legend()
plt.savefig('data/transport_result.png', bbox_inches='tight')
plt.show()
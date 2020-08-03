import numpy as np
import scipy.io
import scipy.linalg


def Plot_GP(plt, X, mu, K, color, mean_alpha=1, var_alpha=0.5, label=None):

    if label:
        plt.plot(X, mu, c=color, alpha=mean_alpha, label=label)
    else:
        plt.plot(X, mu, c=color, alpha=mean_alpha)

    mu = mu[:, 0]
    s2 = np.diag(K)
    # Notice: There are still issue in the output of Barycenter K
    s = np.sqrt((s2))
    upper = mu + s
    lower = mu - s
    plt.fill_between(X.T[0, :], upper, lower, color=color, alpha=var_alpha)


# Notice: Read the data from original mat file
def read_all_gps(mat_address='data/exampleData.mat'):
    mat = scipy.io.loadmat(mat_address)

    days = mat['days']
    vanavara_gps = mat['Vanavara_GPs']

    num_of_GP = vanavara_gps.shape[1]

    gp_list = []
    for i in range(num_of_GP):
        gp_list.append((vanavara_gps[0, i][0, 0], vanavara_gps[0, i][0, 1]))

    return gp_list, days
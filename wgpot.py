import numpy as np
import scipy.io
import scipy.linalg
# from matplotlib import pyplot as plt


def GP_W_barycenter(gp_list, lbda=None, err=None):
    # Notice: Initialization

    m_gp = len(gp_list)     # Number of GPs
    d_gp = gp_list[0][0].shape[0]   # Dimension of the Gaussians

    means_array = np.zeros((d_gp, m_gp))
    cov_mats = np.zeros((d_gp, d_gp, m_gp))

    for i in range(m_gp):
        means_array[:, i] = gp_list[i][0][:, 0]
        cov_mats[:, :, i] = gp_list[i][1]

    # Notice: Constant limiting the amount of iterations
    uplimit = 10**2

    # Notice:
    #  If error margin is not specified, it's set to 1e-6
    if err is None:
       err = 1e-6
    # Notice if weights are not specified, uniform wrights are chosen
    if lbda is None:
        lbda = (1.0/m_gp) * np.ones((1, m_gp))
        # lbda = (0.0141) * np.ones((1, m_gp))

    # Notice: Iteration
    #  The barycenter is the fixed point of the map F.
    K = cov_mats[:, :, 0]
    K_next = F_map(K, cov_mats, lbda)
    count = 0

    wd = Wasserstein_GP((np.zeros((d_gp, 1)), K), (np.zeros((d_gp, 1)), K_next))

    while wd > err and count < uplimit:
        K = K_next
        K_next = F_map(K, cov_mats, lbda)
        count = count + 1
        print('count =', count)
        wd = Wasserstein_GP((np.zeros((d_gp, 1)), K), (np.zeros((d_gp, 1)), K_next))
        print(' W-d in this iteration =', wd)

    if count == uplimit:
        print('Barycenter did not converge')

    mu_mean = np.sum(np.multiply(np.tile(lbda, (d_gp, 1)), means_array), axis=1, keepdims=1)

    return mu_mean, K_next


# Notice: Squared 2-Wasserstein distance of GPs
def Wasserstein_GP(gp_0, gp_1):

    mu_0 = gp_0[0]
    K_0 = gp_0[1]

    mu_1 = gp_1[0]
    K_1 = gp_1[1]

    sqrtK_0 = scipy.linalg.sqrtm(K_0)
    first_term = np.dot(sqrtK_0, K_1)
    K_0_K_1_K_0 = np.dot(first_term, sqrtK_0)

    cov_dist = np.trace(K_0) + np.trace(K_1) - 2 * np.trace(scipy.linalg.sqrtm(K_0_K_1_K_0))
    l2norm = (np.sum(np.square(abs(mu_0 - mu_1))))
    d = np.real(np.sqrt(l2norm + cov_dist))

    return d


# Notice
#   The covariance matrix of the barycenter is the fixed point of
#   the following map F
def F_map(K, cov_mats, lbda):

    sqrtK = np.real(scipy.linalg.sqrtm(K))
    d_gp = cov_mats.shape[0]
    m_gp = lbda.shape[1]
    T = np.zeros((d_gp, d_gp))

    for i in range(m_gp):
        K_bar_K_i_K_bar = np.dot(np.dot(sqrtK, cov_mats[:, :, i]), sqrtK)
        T = T + lbda[0, i] * np.real(scipy.linalg.sqrtm(K_bar_K_i_K_bar))

    # Notice
    #   x = np.linalg.solve(B.conj().T, A.conj().T).conj().T
    #   https://stackoverflow.com/questions/1007442/mrdivide-function-in-matlab-what-is-it-doing-and-how-can-i-do-it-in-python
    #   x = np.linalg.lstsq(sqrtK.T, np.square(T).T)[0] #
    #   x = np.linalg.solve(sqrtK.T, np.square(T).T)
    #   x = np.dot(np.dot(T, T), np.linalg.inv(sqrtK))

    scd_term = np.linalg.solve(sqrtK.conj().T, np.dot(T, T).conj().T).conj().T
    T = np.linalg.solve(sqrtK, scd_term)
    return T


def logmap(mu_gp1, K_gp1, mu_gp2, K_gp2):
    # Notice: The logarithmic map from GD1 tp GD2 on the Riemannian manifold
    #  of GDs with the W metric, see "W geometry of Gaussian measure"
    #   The logmap.m from [Anton NIPS 2017]

    v_mu = mu_gp1 - mu_gp2
    d_gp = mu_gp1.shape[0]
    # Notice: * Here apply the transport map of Gaussian Process!
    #   Proposition 2 of
    #   "Procrustes Metrics on Covariance Operators and
    #   Optimal Transportation of Gaussian Processes"

    sqrtK2 = np.real(scipy.linalg.sqrtm(K_gp2))
    sqrt_sK2_K1_sK2 = np.real(scipy.linalg.sqrtm(np.dot(np.dot(sqrtK2, K_gp1),sqrtK2)))
    scd_part = np.linalg.solve(sqrt_sK2_K1_sK2, sqrtK2)
    T = np.dot(sqrtK2, scd_part) - np.eye(d_gp)

    return v_mu, T


def expmap(mu_gp1, K_gp1, v_mu_t, v_K_t):

    n = mu_gp1.shape[0]
    # print('n =', n)
    q_mu = mu_gp1 + v_mu_t

    v_eye = np.eye(n) + v_K_t
    q_K = np.dot(v_eye, np.dot(K_gp1, v_eye))

    return q_mu, q_K
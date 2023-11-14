import numpy as np
import ot
import scipy.stats as sps
import scipy.linalg as spl
from scipy.optimize import linprog
import matplotlib.pyplot as plt

#################
### author : Julie Delon
#################

###############################
###### GW2 between GMM
###############################


def GaussianW2(m0,m1,Sigma0,Sigma1):
    # compute the quadratic Wasserstein distance between two Gaussians with means m0 and m1 and covariances Sigma0 and Sigma1
    Sigma00  = spl.sqrtm(Sigma0)
    Sigma010 = spl.sqrtm(Sigma00@Sigma1@Sigma00)
    d        = np.linalg.norm(m0-m1)**2+np.trace(Sigma0+Sigma1-2*Sigma010)
    return d


def GW2(pi0,pi1,mu0,mu1,S0,S1):
    # return the GW2 discrete map and the GW2 distance between two GMM
    K0 = mu0.shape[0]
    K1 = mu1.shape[0]
    d  = mu0.shape[1]
    S0 = S0.reshape(K0,d,d)
    S1 = S1.reshape(K1,d,d)
    M  = np.zeros((K0,K1))
    # First we compute the distance matrix between all Gaussians pairwise
    for k in range(K0):
        for l in range(K1):
            M[k,l]  = GaussianW2(mu0[k,:],mu1[l,:],S0[k,:,:],S1[l,:,:])
    # Then we compute the OT distance or OT map thanks to the OT library
    wstar     = ot.emd(pi0,pi1,M)         # discrete transport plan
    distGW2   = np.sum(wstar*M)
    return wstar,distGW2


if __name__ == '__main__':
    # first GMM
    d = 1  # space dimension

    # pi0 = np.array([1.])
    # mu0 = np.array([[.4]]).T
    # S0 = np.array([[[.0016]]])

    pi0 = np.array([.3, .7])
    mu0 = np.array([[.2, .4]]).T
    S0 = np.array([[[.0009]], [[.0016]]])

    # second GMM
    pi1 = np.array([.6, .4])
    mu1 = np.array([[.6, .8]]).T
    S1 = np.array([[[.0036]], [[.0049]]])

    wstar,dist = GW2(pi0,pi1,mu0,mu1,S0,S1)
    wstar
# Paper: Zamani, B., Kennedy, J., Chapman, A., Dower, P., Manzie, C., & Crase, S. (2025).
# GMM-Based Time-Varying Coverage Control. arXiv preprint arXiv:2507.18938.

# The code below is based off the equations in the paper, relating to the simulated plume in Fig. 1
# This script defines a 2D, 5-component Gaussian Mixture Model (GMM) whose component means move linearly
# between six keyframe configurations S0–S5 over 150 seconds.

import numpy as np

# --- constant parameters from paper ---
n_components = 5
sigma = 15.0  # [m]
amplitude = 100.0  # unused here (density scaling)
time_keyframes = np.array([0, 30, 60, 90, 120, 150])  # [s]

# --- keyframe positions ---
S = [
    np.array([[30, 55, 85, 100, 110],
              [15, 25, 55, 22, 35]]),  # S0
    np.array([[50, 65, 90, 110, 120],
              [20, 35, 65, 42, 45]]),  # S1
    np.array([[70, 95, 105, 110, 130],
              [40, 45, 65, 62, 55]]),  # S2
    np.array([[90, 115, 125, 130, 145],
              [20, 55, 75, 62, 65]]),  # S3
    np.array([[110, 130, 135, 140, 150],
              [30, 60, 80, 85, 75]]),  # S4
    np.array([[140, 145, 150, 150, 175],
              [35, 70, 92, 95, 60]])   # S5
]

def get_source_positions(t):
    """
    Interpolation function
    """
    if t <= time_keyframes[0]:
        return S[0]
    if t >= time_keyframes[-1]:
        return S[-1]

    l = np.searchsorted(time_keyframes, t) - 1
    alpha = (t - time_keyframes[l]) / (time_keyframes[l + 1] - time_keyframes[l])
    return S[l] + alpha * (S[l + 1] - S[l])

def get_gmm_parameters(t):
    """
    GMM parameter function
    """
    means = get_source_positions(t).T  # (5,2)
    covs = np.array([np.eye(2) * (sigma ** 2) for _ in range(n_components)])  # (5,2,2)
    weights = np.ones(n_components) / n_components
    return {
        "time": t,
        "means": means,
        "covariances": covs,
        "weights": weights
    }

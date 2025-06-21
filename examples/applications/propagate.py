import torch
import discretize_distributions as dd

import discretize_distributions.schemes as dd_schemes
import discretize_distributions.distributions as dd_dists
import discretize_distributions.optimal as dd_optimal

from matplotlib import pyplot as plt
import numpy as np
from dynamics import DubinsCar
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def dubins_car_ode(t, state, v0=0.15, w=0.1):
    x, y, theta = state
    dxdt = v0 * np.cos(theta)
    dydt = v0 * np.sin(theta)
    dthetadt = w
    return [dxdt, dydt, dthetadt]

# settings
dt = 1
T = 10
v = 1.5
num_locs = 100
dubins_car = DubinsCar(v=v, dt=dt)

loc = torch.tensor([0.0, 0.0, 0.0])
covariance_matrix = torch.diag_embed(torch.tensor([[1.0, 1.0, 1.0]]))
component_distribution = dd_dists.MultivariateNormal(loc=loc, covariance_matrix=covariance_matrix)
mixture_distribution = torch.distributions.Categorical(probs=torch.tensor([1,]))

x_0 = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

centers, clusters = dd_optimal.dbscan_clusters(gmm=x_0)
mix_grid0 = dd_optimal.create_grid_from_clusters(x_0, centers, clusters, num_locs=num_locs)
disc0, _ = dd.discretize(x_0, mix_grid0)

x_k = disc0

theta0 = 0.0
omega = 0.1
u = torch.tensor([omega])
num_particles = x_k.locs.shape[0]
timesteps = int(T/dubins_car.dt)
trajectories = []

# emp approx
M = 500
empirical_samples = x_0.sample((M,))
empirical_trajectories = []
empirical_state = empirical_samples  # (M, 3)

for k in range(timesteps):
    trajectories.append(x_k.locs.detach().clone())
    empirical_trajectories.append(empirical_state.detach().clone())
    # dynamics using signature
    u_batched = u.repeat(len(x_k.locs), 1)
    x_next = dubins_car.rk4_step(x_k.locs, u, dubins_car.dt)  # no noise
    print(f'Size of disc: {len(x_next)}')

    # emp approx dynamics
    u_emp = u.repeat(empirical_state.shape[0], 1)
    empirical_state = dubins_car.rk4_step(empirical_state, u_emp, dubins_car.dt)
    print(f'Size of emp: {len(empirical_state)}')

    # new categorical float
    disc = dd_dists.CategoricalFloat(locs=x_next, probs=x_k.probs)
    x_k = disc

    #################### WITH NOISE ###########################
    # locs are the propagated locs from previous step, prob mass from previous step, cov stays same due to
    # additive noise - write out and prove later
    # component_distribution = dd_dists.MultivariateNormal(loc=x_next, covariance_matrix=covariance_matrix)
    # mixture_distribution = torch.distributions.Categorical(probs=x_k.probs)
    # dist = dd_dists.MixtureMultivariateNormal(mixture_distribution, component_distribution)

    # discretize again - when we have noise
    # centers, clusters = dd_optimal.dbscan_clusters(gmm=dist)
    # mix_grid = dd_optimal.create_grid_from_clusters(dist, centers, clusters, num_locs=num_locs)
    # disc, _ = dd.discretize(dist, mix_grid)
    # x_k = disc

x0, y0, theta0 = 0.0, 0.0, 0.0
state0 = [x0, y0, theta0]
t_eval = np.arange(0, T, dt)
solution = solve_ivp(dubins_car_ode, [0, T], state0, t_eval=t_eval, args=(v, dt))
x_traj, y_traj, theta_traj = solution.y

### 2D plot
# ODE reference
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(x_traj, y_traj, label="ODE Trajectory", color="blue", linewidth=2)

# Discretized
for k, locs in enumerate(trajectories):
    x, y = locs[:, 0].numpy(), locs[:, 1].numpy()
    ax.scatter(x, y, color='red', alpha=0.3, s=10, label="Discretized" if k == 0 else None)

# Empirical
for k, locs in enumerate(empirical_trajectories):
    x, y = locs[:, 0].numpy(), locs[:, 1].numpy()
    ax.scatter(x, y, color='green', alpha=0.3, s=10, label="Empirical" if k == 0 else None)

ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Dubins Car: Discretized vs Empirical vs ODE")
ax.legend()
ax.grid()
plt.axis("equal")
plt.show()


##### 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# ODE
ax.plot(solution.y[0], solution.y[1], solution.y[2], label="ODE Trajectory", color='blue', linewidth=2)

# empirical particles
for k, locs in enumerate(empirical_trajectories):
    x, y, theta = locs[:, 0].numpy(), locs[:, 1].numpy(), locs[:, 2].numpy()
    ax.scatter(x, y, theta, color='green', alpha=0.05, s=10, label='Empirical' if k == 0 else None)

# discretized particles
# same color
for k, locs in enumerate(trajectories):
    x, y, theta = locs[:, 0].numpy(), locs[:, 1].numpy(), locs[:, 2].numpy()
    ax.scatter(x, y, theta, color='red', alpha=0.3, s=10, label='Discretized' if k == 0 else None)

# each time step diff colors
# cmap = cm.get_cmap('viridis', len(trajectories))
#
# for k, locs in enumerate(trajectories):
#     x = locs[:, 0].numpy()
#     y = locs[:, 1].numpy()
#     theta = locs[:, 2].numpy()
#
#     color = cmap(k / len(trajectories))  # Normalize k to [0,1] for the colormap
#     ax.scatter(x, y, theta, color=color, alpha=0.5, s=10)

ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Theta (rad)")
ax.set_title("Dubins Car Trajectories in (x, y, θ) Space")
ax.legend()
plt.tight_layout()
plt.show()
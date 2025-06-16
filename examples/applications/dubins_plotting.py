import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dynamics import DubinsCar

# Define the Dubins car dynamics
def dubins_car_ode(t, state, v0=0.15, w=0.1):
    x, y, theta = state
    dxdt = v0 * np.cos(theta)
    dydt = v0 * np.sin(theta)
    dthetadt = w
    return [dxdt, dydt, dthetadt]

# Initial conditions
x0, y0, theta0 = 0.0, 0.0, 0.0  # Initial state (x, y, theta)
state0 = [x0, y0, theta0]

# Simulation time
T = 10.0  # Total simulation time
dt = 0.05  # Time step
t_eval = np.arange(0, T, dt)

# Solve the ODE using scipy's solve_ivp
solution = solve_ivp(dubins_car_ode, [0, T], state0, t_eval=t_eval, args=(0.15, 0.1))

# Extract results
x_traj, y_traj, theta_traj = solution.y

# Plot the trajectory in the x-y plane
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x_traj, y_traj, label="Dubins Car Trajectory", color="b")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Dubins Car Simulation Using ODEs")
ax.legend()
ax.grid()

plt.show()

# Define the grid
x_vals = np.linspace(-5, 5, 20)
y_vals = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x_vals, y_vals)

# Fix theta = 0 rad
# theta0 = -5.0
# U = np.zeros_like(X)
# V = np.zeros_like(Y)

v0 = 0.15
theta_grid = np.arctan2(Y, X)
U = v0 * np.cos(theta_grid)
V = v0 * np.sin(theta_grid)


# Evaluate vector field
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        state = [X[i, j], Y[i, j], theta_grid[i, j]]
        dxdt, dydt, _ = dubins_car_ode(0, state)
        U[i, j] = dxdt
        V[i, j] = dydt

# Plot
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, color='blue', angles='xy', scale_units='xy', scale=0.3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Continuous Dubins Car Vector Field')
plt.grid()
plt.axis('equal')
plt.show()

# Discrete dynamics vector fields (IS it possible? )
dubins_car = DubinsCar(v=1.5, dt=0.1)

# Define grid
x_vals = np.linspace(-5, 5, 20)
y_vals = np.linspace(-5, 5, 20)
X, Y = np.meshgrid(x_vals, y_vals)

# Assume theta = 0 and omega = 0.1 for all grid points
theta0 = 0.0
omega = 0.1
U = np.zeros_like(X)
V = np.zeros_like(Y)

# Evaluate state transitions
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        state = torch.tensor([X[i, j], Y[i, j], theta0])
        u = torch.tensor([omega])
        next_state = dubins_car.rk4_step(state, u, dubins_car.dt, noise_std=0.05)
        dx = next_state[0, 0] - X[i, j]
        dy = next_state[0, 1] - Y[i, j]
        U[i, j] = dx
        V[i, j] = dy

# Plot
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, color='green', angles='xy', scale_units='xy', scale=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Dubins Car Discrete Dynamics Vector Field')
plt.grid()
plt.axis('equal')
plt.show()

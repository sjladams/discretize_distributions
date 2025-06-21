import torch
from abc import ABC
import matplotlib.pyplot as plt
import numpy as np


class _Dynamics(ABC):
    def __call__(self, *args, **kwargs):
        pass


class LinearDynamics(_Dynamics):
    def __init__(self, A: torch.Tensor, B: torch.Tensor):
        self.A = A
        self.B = B

    def __call__(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.A.T) + torch.matmul(u, self.B.T)


class NonlinearDynamics(_Dynamics):
    def __init__(self, dynamics_function):
        self.dynamics_function = dynamics_function  # f(x, u)

    def __call__(self, x: torch.Tensor, u: torch.Tensor, dt: float = 0.01) -> torch.Tensor:
        dx = self.dynamics_function(x, u)
        return x + dx * dt


class DubinsCar:
    def __init__(self, v=1.5, dt=0.1, init_mean=None, init_cov=None, target=None):
        self.dt = dt
        self.v = v
        self.state_dim = 3
        self.control_dim = 1
        self.init_mean = init_mean if init_mean is not None else torch.tensor([0.0, 0.0, 0.0])
        self.init_cov = init_cov if init_cov is not None else 0.01 * torch.eye(self.state_dim)
        self.target = target if target is not None else torch.tensor([11.0, 9.0, 0.0])

    def __call__(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.rk4_step(x, u, self.dt)

    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        x: (N, D), u: (N,) or (N, 1)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if u.dim() == 1:
            u = u.unsqueeze(1)

        theta = x[:, 2]
        dx_dt = torch.zeros_like(x)
        dx_dt[:, 0] = self.v * torch.cos(theta)
        dx_dt[:, 1] = self.v * torch.sin(theta)
        dx_dt[:, 2] = u.squeeze(-1)  # omega

        return dx_dt

    def rk4_step(self, x: torch.Tensor, u: torch.Tensor, dt: float, noise_std=0.01) -> torch.Tensor:
        k1 = self.dynamics(x, u)
        k2 = self.dynamics(x + 0.5 * dt * k1, u)
        k3 = self.dynamics(x + 0.5 * dt * k2, u)
        k4 = self.dynamics(x + dt * k3, u)

        x_next = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Gaussian noise
        # noise = torch.randn_like(x_next) * noise_std
        # x_next += noise

        return x_next

    def rollout_rk4(self, x: torch.Tensor, u_seq: torch.Tensor, N: int) -> torch.Tensor:
        """
        Rollout the system over N steps with a constant control input per step.
        x: (N_particles, D)
        u_seq: (N_steps, control_dim)
        Returns: x after N RK4 steps
        """
        for k in range(N):
            u_k = u_seq[k].expand(x.shape[0], -1)
            x = self.rk4_step(x, u_k, self.dt)
        return x


class DynamicBicycleModel:
    def __init__(self, dt=0.1, params=None, init_mean=None, init_cov=None, target=None):
        self.dt = dt
        self.state_dim = 6
        self.control_dim = 2
        self.init_mean = init_mean if init_mean is not None else torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.init_cov = init_cov if init_cov is not None else 0.001 * torch.eye(self.state_dim)
        self.target = target if target is not None else torch.tensor([75.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        # Vehicle parameters
        default_params = {
            "m": 1412.0,
            "Iz": 1536.7,
            "lf": 1.06,
            "lr": 1.85,
            "kf": -128916.0,
            "kr": -85944.0,
        }
        self.params = default_params if params is None else params

    def __call__(self, x, u):
        return self.step(x, u)

    def step(self, x, u):
        m, Iz, lf, lr, kf, kr = [self.params[k] for k in ("m", "Iz", "lf", "lr", "kf", "kr")]
        dt = self.dt

        # Split state
        x_pos, y_pos, phi, u_long, v_lat, omega = torch.unbind(x, dim=-1)
        # print("DEBUG: u.shape =", u.shape)
        a, delta = torch.unbind(u, dim=-1)

        # Lateral tire forces using simplified linear model
        Fy1 = kf * ((v_lat + lf * omega) / (u_long + 1e-5) - delta)
        Fy2 = kr * ((v_lat - lr * omega) / (u_long + 1e-5))

        # Update dynamics
        x_pos_next = x_pos + dt * (u_long * torch.cos(phi) - v_lat * torch.sin(phi))
        y_pos_next = y_pos + dt * (v_lat * torch.cos(phi) + u_long * torch.sin(phi))
        phi_next = phi + dt * omega
        u_next = u_long + dt * a
        v_next = (m * u_long * v_lat + dt * (lf * kf - lr * kr) * omega - dt * kf * delta * u_long - dt * m * u_long**2
                  * omega) / (m * u_long - dt * (kf + kr) + 1e-5)
        omega_next = ((Iz * u_long * omega + dt * (lf * kf - lr * kr) * v_lat - dt * lf * kf * delta * u_long)
                      / (Iz * u_long - dt * (lf**2 * kf + lr**2 * kr) + 1e-5))

        return torch.stack([x_pos_next, y_pos_next, phi_next, u_next, v_next, omega_next], dim=-1)

    def rk4_step(self, x: torch.Tensor, u: torch.Tensor, dt: float) -> torch.Tensor:
        def f(x, u):
            return self.step(x, u)

        k1 = f(x, u)
        k2 = f(x + 0.5 * dt * k1, u)
        k3 = f(x + 0.5 * dt * k2, u)
        k4 = f(x + dt * k3, u)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


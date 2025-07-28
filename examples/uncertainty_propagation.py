from typing import Union, Callable, Optional

import torch
import math
import ot
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Circle

import discretize_distributions as dd
import discretize_distributions.distributions as dd_dists
import discretize_distributions.generate_scheme as dd_gen

COLORS = ['Blues', 'BuPu', 'PuRd', 'Greens', 'Oranges', 'Reds', 'Greys', 'Purples',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

class Dynamics:
    def __init__(self, global_lipschitz: float):
        self.global_lipschitz = global_lipschitz

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.global_lipschitz

def propagate_state_dist_over_dynamics(
        dynamics: Dynamics,
        noise_dist: dd_dists.MultivariateNormal,
        sign_state_dist: dd_dists.CategoricalFloat
):
    assert isinstance(noise_dist, dd_dists.MultivariateNormal)
    sign_q = sign_state_dist  
    q1 = dd_dists.MixtureMultivariateNormal(
        mixture_distribution=torch.distributions.Categorical(
            probs=sign_state_dist.probs),
        component_distribution=dd_dists.MultivariateNormal(
            loc=dynamics(sign_state_dist.locs) + noise_dist.loc,
            covariance_matrix=noise_dist.covariance_matrix
        ))
    return sign_q, q1


def single_step(
        dynamics: Dynamics,  
        noise_dist: dd_dists.MultivariateNormal,
        q: Union[dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal],
        num_samples: int,
        grid_size: int,
        w2_p__q_global_lipschitz: float = 0.,
        run_empirical: bool = False,
        p_samples: Optional[torch.Tensor] = None,
):
    # Approximate the state distribution
    scheme = dd_gen.generate_scheme(
        q, 
        grid_size=grid_size, 
        prune_factor=0.01, 
        n_iter=1000,
        lr=0.01
    )
    sign_q, w2_q__disc_q = dd.discretize(q, scheme)

    if isinstance(sign_q, dd_dists.CategoricalGrid):
        sign_q = sign_q.to_categorical_float()

    # Propagate the (approximate) state distribution over the dynamics
    sign_q, q1 = propagate_state_dist_over_dynamics(dynamics, noise_dist, sign_q)

    # Empirically approximate the state distribution
    q_samples = q.sample(torch.Size((num_samples,)))
    q1_samples = q1.sample(torch.Size((num_samples,)))
    noise_samples = noise_dist.sample(torch.Size((num_samples,)))

    p1_samples = dynamics(p_samples if p_samples is not None else q_samples) + noise_samples

    # Propagate wasserstein error, i.e., compute W_2(p_1, q_1) = W_2(f#p_k, f#\Delta_C#q_k)
    w2_p1__q1_global_lipschitz =dynamics.global_lipschitz * (w2_q__disc_q + w2_p__q_global_lipschitz)

    if run_empirical:
        w2_p1__q1_empirical = ot.solve_sample(p1_samples.view(-1, p1_samples.shape[-1]),
                                                 q1_samples.view(-1, q1_samples.shape[-1])
                                                 ).value.sqrt()
    else:
        w2_p1__q1_empirical = torch.nan

    return dict(
        w2_q__sign_q=w2_q__disc_q,
        w2_p1__q1_empirical=w2_p1__q1_empirical,
        w2_p1__q1_global_lipschitz=w2_p1__q1_global_lipschitz,
        q1=q1,
        q=q,
        q1_samples=q1_samples,
        p1_samples=p1_samples
    )


def multi_step(
    dynamics: Dynamics,
    noise_dist: dd_dists.MultivariateNormal,
    q: Union[dd_dists.MultivariateNormal, dd_dists.MixtureMultivariateNormal],
    num_time_steps: int,
    num_samples: int,
    grid_size: int,
    run_empirical: bool = False
):
    w2_p1__q1_store = {-1: dict(w2_p1__q1_global_lipschitz=0.)}
    w2_q__sign_q_store = dict()
    q_store = {-1: {'q1': q}}

    # initialize empirical distributions
    samples_store = {-1: {'p1_samples': q.sample(torch.Size((num_samples,))), 
                          'q1_samples': q.sample(torch.Size((num_samples,)))}}

    # loop over time steps
    for k in range(num_time_steps):
        print(f'---- TIME STEP {k} ----')
        out = single_step(
            dynamics=dynamics,
            noise_dist=noise_dist,
            q=q_store[k-1]['q1'],
            p_samples=samples_store[k-1]['p1_samples'],
            w2_p__q_global_lipschitz=w2_p1__q1_store[k-1]['w2_p1__q1_global_lipschitz'],
            num_samples=num_samples,
            grid_size=grid_size, 
            run_empirical=run_empirical
        )

        w2_p1__q1_store[k] = {key: value for key, value in out.items() if 'w2_p1__q1' in key}
        w2_q__sign_q_store[k] = out['w2_q__sign_q']
        samples_store[k] = {key: value for key, value in out.items() if 'samples' in key}
        q_store[k] = dict(q1=out['q1'], q=out['q'])

        print(
            f"Bounds on W_2(p_{k+1}, q_{k+1}) with len(sup(q_{k+1}))={out['q1'].num_components} via:\n"
            f"\t Global Lipschitz: {out['w2_p1__q1_global_lipschitz']:.4f}\n"
            f"\t Empirical: {out['w2_p1__q1_empirical']:.4f}\n"
        )

    return w2_q__sign_q_store, w2_p1__q1_store, samples_store, q_store



@torch.no_grad()
def plot_2d_dynamics(dynamics,  xlim: Optional[list] = None, ylim: Optional[list] = None, scale: float = 1.0):
    xlim = [-1, 1] if xlim is None else xlim
    ylim = [-1, 1] if ylim is None else ylim

    x = torch.linspace(xlim[0], xlim[1], 5 * int(xlim[1] - xlim[0]))
    y = torch.linspace(ylim[0], ylim[1], 5 * int(ylim[1] - ylim[0]))
    X, Y = torch.meshgrid(x, y, indexing="ij")
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)  # Shape (N, 2)

    with torch.no_grad():
        next_states = dynamics(grid_points)
        deltas = next_states - grid_points

    plt.figure(figsize=(6 * (xlim[1] - xlim[0]), 6 * (ylim[1] - ylim[0])))
    plt.quiver(
        grid_points[:, 0].numpy(),  # X coordinates
        grid_points[:, 1].numpy(),  # Y coordinates
        deltas[:, 0].numpy(),  # U: delta X
        deltas[:, 1].numpy(),  # V: delta Y
        angles='xy', scale_units='xy',
        scale=scale,
        width=0.003
    )
    plt.xlim(xlim) if xlim is not None else None
    plt.ylim(ylim) if ylim is not None else None

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_2d_ambiguity_balls(samples: Union[dict, list], w2_p1__q1_store: Union[dict, list], q_store: Union[dict, list],
                            step_size: int = 1,
                            xlim: Optional[list] = None, ylim: Optional[list] = None):
    xlim = [-1, 1] if xlim is None else xlim
    ylim = [-1, 1] if ylim is None else ylim

    if isinstance(samples, list) and isinstance(w2_p1__q1_store, list) and isinstance(q_store, list):
        samples_options, w2_p1__q1_store_options, q_store_options = samples, w2_p1__q1_store, q_store
    else:
        samples_options, w2_p1__q1_store_options, q_store_options = [samples], [w2_p1__q1_store], [q_store]


    for tag in ['p1_samples', 'q1_samples']:
        fig, ax = plt.subplots(figsize=(6 * (xlim[1] - xlim[0]), 6 * (ylim[1] - ylim[0])))

        for samples, w2_p1__q1_store, q_store in zip(samples_options, w2_p1__q1_store_options, q_store_options):
            time_steps = list(q_store.keys())[::step_size][:-1]
            cmap = plt.cm.coolwarm
            colors = [cmap(i / (len(time_steps) - 1)) for i in range(len(time_steps))]

            for k in time_steps:
                q = q_store[k+1]['q']
                ax.scatter(samples[k][tag][:, 0], samples[k][tag][:, 1], color=colors[k + 1], s=16, alpha=0.5)

                ambiguity_set = Circle(q.mean, w2_p1__q1_store[k]['w2_p1__q1_global_lipschitz'], color=colors[k+1], fill=False, lw=2, alpha=1.0)
                ax.add_patch(ambiguity_set)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        plt.xlim(xlim) if xlim is not None else None
        plt.ylim(ylim) if ylim is not None else None
        plt.tight_layout()
        plt.show()

            

class LinearDynamics(Dynamics):
    def __init__(self, global_lipschitz: float, mat: torch.Tensor):
        super().__init__(global_lipschitz=global_lipschitz)
        self.mat = mat

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return  torch.einsum('ij,...j->...i', self.mat, x)


def rot_mat(theta, rho, delta):
    theta = theta if torch.is_tensor(theta) else torch.as_tensor(theta)
    rho = rho if torch.is_tensor(rho) else torch.as_tensor(rho)
    delta = delta if torch.is_tensor(delta) else torch.as_tensor(delta)
    return rho * torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]]) + delta

if __name__== '__main__':
    mat = rot_mat(theta=-math.pi / 8., rho=0.8, delta=0.)
    dynamics = LinearDynamics(global_lipschitz=torch.linalg.norm(mat, ord=2), mat=mat)

    w2_q__sign_q_store, w2_p1__q1_store, samples_store, q_store = multi_step(
        dynamics=dynamics,
        noise_dist=dd_dists.MultivariateNormal(
            loc=torch.zeros(2),
            covariance_matrix=torch.eye(2) * 0.001
        ),
        q=dd_dists.MultivariateNormal(
            loc=torch.ones(2) * 0.8,
            covariance_matrix=torch.eye(2) * 0.0001
        ),
        num_time_steps=3,
        num_samples=100,
        grid_size=10, 
        run_empirical=False
    )

    xlim, ylim = [-1., 1.], [-1., 1.]
    plot_2d_dynamics(dynamics, xlim=xlim, ylim=ylim)
    plot_2d_ambiguity_balls(samples_store, w2_p1__q1_store, q_store, xlim=xlim, ylim=ylim)
    

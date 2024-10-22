import torch
from torch_kmeans import KMeans
from tqdm import tqdm
import os

from .utils import pickle_dump, cdf, calculate_w2_disc_uni_stand_normal, calculate_mean_and_var_trunc_normal, get_edges
from .tensors import symmetrize_vector


def find_locs(nr_locs: int, locs_init: torch.Tensor = None,
            nr_iterations: int = 1000, lr: float = 0.001, early_stop: bool = True,
            plot_loss: bool = False, show_progress: bool = False):

    # setup
    if locs_init is None:
        # use initialization as proposed in "Optimal quadratic quantization for numerics: The Gaussian case"
        locs_init = torch.tensor([-2 + 2*(2*k-1)/nr_locs for k in range(1, nr_locs+1)])
    else:
        locs_init = locs_init.detach().clone()

    uneven_nr_locs = nr_locs % 2 == 1
    locs = locs_init.sort().values
    locs.requires_grad_(True)

    # optimization
    optimizer = torch.optim.Adam([locs, ])
    losses = list()

    iterations = range(nr_iterations * nr_locs)
    if show_progress:
        iterations = tqdm(iterations)

    for i in iterations:
        # enforce symmetry:
        symmetric_locs = symmetrize_vector(locs)

        loss = calculate_w2_disc_uni_stand_normal(symmetric_locs)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.detach())

        if early_stop and i > 50 and abs(losses[i] - losses[i - 50]) < 1e-6:
            break

    losses = torch.tensor(losses)
    locs = locs.detach().sort().values
    locs = symmetrize_vector(locs)

    return locs, losses


def generate_lookup_opt_grid_uni_std_normal(tag:str, max_num_locs: int, random_init: bool, num_samples: int,
                                        opt_params: dict):
    assert num_samples > max_num_locs, ("Number of samples for empirical validation and initialization should be "
                                       "larger than number of signature locations")

    path_to_lookup_opt_grid_uni_norm = f".{os.sep}data{os.sep}{tag}"

    # generate samples for empirical initialization and validation
    num_samples = max(1000, max_num_locs)
    norm = torch.distributions.Normal(loc=torch.zeros(1), scale=torch.ones(1))
    samples = norm.sample((num_samples,))

    # generate table for 2:n points
    store = {'locs': dict(),
             'probs': dict(), 'trunc_mean': dict(), 'trunc_var': dict(), 'w2': dict(),
             'locs_emp': dict(), 'w2_emp': dict()}
    store_loss_traj = dict()

    # loop over size options
    for nr_locs in torch.arange(1, max_num_locs + 1):
        uneven_nr_locs = nr_locs % 2 == 1

        if nr_locs == 1:
            locs = torch.zeros(1)
            locs_emp = locs
            w2_emp = samples.pow(2).sum() * (1 / num_samples)
        elif uneven_nr_locs:
            kmeans_torch = KMeans(n_clusters=int(nr_locs))
            kmeans_result = kmeans_torch(samples.unsqueeze(0))
            locs_emp = kmeans_result.centers.squeeze().sort().values
            # TO compute the squared 2-Wasserstein distance use the inertia, which is the sum of squared distances of
            # samples to their closest cluster center
            w2_emp = (kmeans_result.inertia.squeeze() * (1 / num_samples))

            if random_init:
                # Enforce symmetry of empirical estimation
                locs_init = symmetrize_vector(locs_emp)
            else:
                locs_init = None

            locs, loss_traj = find_locs(nr_locs, locs_init=locs_init, **opt_params)
            store_loss_traj[nr_locs] = loss_traj
        elif nr_locs == 2:
            # Optimal strategy is known to be tw
            locs = torch.Tensor(norm.log_prob(torch.zeros(1)).exp() * 2)
            locs = torch.cat((-locs, locs))
            locs_emp = locs
            # Compute empirical wasserstein via half plane (prob = 0.5)
            w2_emp = (samples.abs() - locs[-1]).pow(2).sum() * (0.5 / num_samples) * 2
        else:
            # Use symmetry for efficient sampling
            kmeans_torch = KMeans(n_clusters=int(nr_locs) // 2)
            kmeans_result = kmeans_torch(samples.abs().unsqueeze(0))
            locs_emp = kmeans_result.centers.squeeze().sort().values
            locs_emp = torch.cat((-locs_emp.flip(0), locs_emp))
            # Compute empirical wasserstein via half plane (prob = 0.5)
            w2_emp = (kmeans_result.inertia.squeeze() * (0.5 / num_samples)) * 2

            # formal
            locs, loss_traj = find_locs(nr_locs, locs_init=locs_emp, **opt_params)
            store_loss_traj[nr_locs] = loss_traj

        assert torch.equal(locs, locs.sort().values), "locs should be stored sorted"

        edges = get_edges(locs)
        probs = cdf(edges[1:]) - cdf(edges[0:-1])
        trunc_mean, trunc_var = calculate_mean_and_var_trunc_normal(loc=0., scale=1., l=edges[:-1], u=edges[1:])
        w2 = calculate_w2_disc_uni_stand_normal(locs)

        store['locs'][int(nr_locs)] = locs
        store['w2'][int(nr_locs)] = w2
        store['probs'][int(nr_locs)] = probs
        store['trunc_mean'][int(nr_locs)] = trunc_mean
        store['trunc_var'][int(nr_locs)] = trunc_var
        store['locs_emp'][int(nr_locs)] = locs_emp
        store['w2_emp'][int(nr_locs)] = w2_emp

        print("nr_points: {}, w2: {:.4f} / {:.4f}".format(nr_locs, w2, w2_emp))

    pickle_dump(store, path_to_lookup_opt_grid_uni_norm)
    pickle_dump(store_loss_traj, f".{os.sep}losses")


if __name__ == "__main__":
    torch.manual_seed(1)

    generate_lookup_opt_grid_uni_std_normal(
        tag='lookup_opt_grid_uni_normal_NEW',
        max_num_locs=300,
        random_init=True,
        num_samples=501,
        opt_params={
            'nr_iterations': 5000,
            'lr': 0.1,
            'plot_loss': False}
    )



from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
import time

import csv
import torch

import discretize_distributions as dd
import discretize_distributions.distributions as dd_dists

import pickle

def pickle_load(tag):
    if not ".pickle" in tag:
        tag = f"{tag}.pickle"
    pickle_in = open(tag, "rb")
    to_return = pickle.load(pickle_in)
    pickle_in.close()
    return to_return

@dataclass
class Benchmarks:
    data: Dict[str, dd_dists.MixtureMultivariateNormal] = field(default_factory=dict)

    def append(self, key: str, locs: torch.Tensor, covs: torch.Tensor, probs: torch.Tensor, num_modes: int):
        mix = torch.distributions.Categorical(probs=probs)
        comp = dd_dists.MultivariateNormal(loc=locs, covariance_matrix=covs)
        gmm = dd_dists.MixtureMultivariateNormal(mix, comp)
        gmm.num_modes = num_modes
        self.data[key] = gmm


    def at(self, key: str) -> dd_dists.MixtureMultivariateNormal:
        return self.data[key]

    def keys(self) -> List[str]:
        return list(self.data.keys())

@dataclass
class Stats:
    w2_error: float
    support_size: int # realized support size
    time: float

@dataclass
class Row: # key is requrest support size
    data: Dict[int, Stats] = field(default_factory=dict)

    def append(self, key: int, rec: Stats):
        self.data[key] = rec

    def keys(self) -> List[int]:
        return list(self.data.keys())

@dataclass
class Table:
    data : Dict[str, Row] = field(default_factory=dict)

    def at(self, key: str) -> Row:
        return self.data[key]

    def keys(self) -> List[str]:
        return list(self.data.keys())
    
    def append(self, key: str, row: Row):
        self.data[key] = row


def generate_csv(
    benchmarks: Benchmarks,
    table: Table,
):
    rows = []
    for name in benchmarks.keys():
        row = dict(
            name=name, 
            num_components=benchmarks.at(name).num_components,
            num_dims=benchmarks.at(name).event_shape[0],
            num_modes=benchmarks.at(name).num_modes
        ) 

        for scheme_size in table.at(name).keys():
            row[f"w2/N={scheme_size}"] = table.at(name).data[scheme_size].w2_error

        for scheme_size in table.at(name).keys():
            row[f"time/N={scheme_size}"] = table.at(name).data[scheme_size].time

        rows.append(row)

    # Write to CSV
    csv_path = os.path.join(os.path.dirname(__file__), "table.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter=';')
        writer.writeheader()
        writer.writerows(rows)


def random_pd_mat(d: int, batch_shape=(), eps: float = 1e-3) -> torch.Tensor:
    A = torch.randn(*batch_shape, d, d)
    pd = A @ A.transpose(-1, -2)
    pd = pd + eps * torch.eye(d)
    return pd

if __name__ == "__main__":
    torch.manual_seed(0)

    benchmarks = Benchmarks()
    # Assa, A., & Plataniotis, K. N. (2018). Wasserstein-distance-based Gaussian mixture reduction. - Figure 2
    benchmarks.append(
        "assa2018wasserstein", 
        locs=torch.tensor([1.45, 2.20, 0.67, 0.48, 1.49, 0.91, 1.01, 1.42, 2.77, 0.89]).view(-1, 1),
        covs=torch.tensor([0.0487, 0.0305, 0.1171, 0.0174, 0.0295, 0.0102, 0.0323, 0.0380, 0.0115, 0.0679]).view(-1,1,1),
        probs=torch.tensor([0.03, 0.18, 0.12, 0.19, 0.02, 0.16, 0.06, 0.10, 0.08, 0.06]),
        num_modes=5
    )
    # Xu, L. & Korba, A. & Slepcev, D. (2022). Accurate Quantization of Measures via Interacting Particle-based Optimization - Figure 10
    benchmarks.append(
        "xu2022accurate", 
        locs=torch.tensor([[1., 0.], [-1., 0.]]),
        covs=torch.diag_embed(torch.ones(2,2)) * 0.3,
        probs=torch.tensor([0.5, 0.5]),
        num_modes=2
    )
    # Terejanu, G. & Singla, P. & Singh, T. & Scott, P. D. (2008). Uncertainty Propagation for Nonlinear Dynamic Systems Using Gaussian Mixture Models
    benchmarks.append(
        "terejanu2008uncertainty", 
        locs=torch.tensor([-0.5, 0.1]).view(-1, 1),
        covs=torch.tensor([0.1, 1.]).view(-1, 1, 1),
        probs=torch.tensor([0.1, 0.9]),
        num_modes=2
    )
    # Runnalls & Andrew R (2007). Kullback-Leibler approach to Gaussian mixture reduction
    benchmarks.append(
        "runnalls2007kullback-example 4.1", 
        locs=torch.tensor([[0., 0.], [0.0001, 0.0001], [-0.0001, -0.0001]]),
        covs=torch.tensor([[[1., 0.9], [0.9, 1.]], [[1., 0.9], [0.9, 1.]], [[1., -0.9], [-0.9, 1.]]]),
        probs=torch.ones(3),
        num_modes=1
    )
    benchmarks.append(
        "runnalls2007kullback-example 4.2", 
        locs=torch.tensor([[0.661, 1.], [1.339, -1.], [-0.692, 1.1], [-1.308, -1.1]]),
        covs=torch.diag_embed(torch.ones(4,2)),
        probs=torch.ones(4),
        num_modes=4
    )
    # Adams S. & Figueiredo E. & L. Laurenti (2025) Formal Uncertainty Propagation for Stochastic Dynamical Systems with Additive Noise - Double Spiral time step 8
    benchmarks.append(
        "double-spiral", 
        locs = torch.tensor([[ 0.5111,  0.4806], [ 0.5192,  0.5002], [ 0.5297,  0.4729], [ 0.5378,  0.4925], [-0.5288,  0.4708], [-0.5348,  0.4853], [-0.5108,  0.4782], [-0.5168,  0.4928]]),
        covs = torch.ones((8, 2, 2)) * 1e-4,
        probs = torch.tensor([0.0782, 0.0568, 0.0868, 0.0608, 0.2060, 0.1504, 0.1941, 0.1670]),
        num_modes=2
    )
    # State after first linear layer of a trained BNN, example from the paper: Adams2024
    bnn_layer_data = pickle_load(os.path.join(os.path.dirname(__file__), "bnn_layer_data"))
    benchmarks.append(
        "BNN Layer", 
        locs=bnn_layer_data["mean"].view(1, -1),
        covs=torch.diag(bnn_layer_data["stddev"]**2).unsqueeze(0),
        probs=torch.ones(1), 
        num_modes=1
    )
    # Degenerative 
    benchmarks.append(
        "degenerative", 
        locs = torch.tensor([[-2.01, -2.01], [-1.99, -1.99], [1.99, 1.99], [2.01, 2.01]]),
        covs = torch.ones((4, 2, 2)),
        probs = torch.tensor([0.25, 0.25, 0.25, 0.25]),
        num_modes=2
    )

    per_mode = True

    scheme_size_options = [10, 100, 1000, 10000]

    table = Table()
    for name in benchmarks.keys():
        gmm = benchmarks.at(name)
        row = Row()
        for scheme_size in scheme_size_options:
            try:
                start_time = time.time()
                scheme = dd.generate_scheme(gmm, scheme_size=scheme_size, per_mode=per_mode, use_analytical_hessian=False)
                disc, w2 = dd.discretize(gmm, scheme)
                stats = Stats(w2_error=w2.item(), support_size=disc.num_components, time=time.time() - start_time)

                row.append(scheme_size, stats)
            except:
                pass
        
        table.append(name, row)

    generate_csv(benchmarks, table)

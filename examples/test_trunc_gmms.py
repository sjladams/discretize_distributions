import discretize_distributions.utils as utils
import torch


def compute_mean_var_trunc_normal(l, u):
    pi = utils.cdf(u) - utils.cdf(l)
    mean = 1 / pi * (utils.pdf(l) - utils.pdf(u))
    var = 1 + 1 / pi * (utils.pdf(l) * l - utils.pdf(u) * u) - mean.pow(2)
    return mean, var

if __name__ == "__main__":
    num_points = 10

    u = torch.randn(num_points)
    l = u - torch.rand(num_points) * 0.5
    assert (u >= l).all(), "Upper bound must be greater than lower bound"

    mean, var = utils.compute_mean_var_trunc_norm(l, u)
    mean_ref, var_ref = compute_mean_var_trunc_normal(l, u)

    print(f"mean: {mean}\n mean_ref: {mean_ref}\n")
    print(f"var: {var}\n var_ref: {var_ref}\n")
    # if not torch.allclose(mean, mean_ref, atol=1e-3):
    #     raise ValueError(f"Mean mismatch: {mean} vs {mean_ref}")
    # if not torch.allclose(var, var_ref, atol=1e-6):
    #     raise ValueError(f"Variance mismatch: {var} vs {var_ref}")


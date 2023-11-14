class DiscretizedSparseMultivariateNormal_(distributions.CategoricalFloat):
    """
    Base class for discretizations (in the form of categoricalFloat objects) of sparse continuous multivariate
    Gaussian distributions
    """
    def __init__(self, norm: distributions.SparseMultivariateNormal, discr_points_per_dim: int = 1, **kwargs):
        if not isinstance(norm, distributions.SparseMultivariateNormal):
            raise ValueError('distribution not of type SparseMultivariateNormal')
        if discr_points_per_dim == 1:
            locs = norm.nonsparse_loc.unsqueeze(-2)
            probs = torch.ones(locs.shape[:-2] + (1,))
        else:
            # eigvals_block, eigvectors_block = torch.linalg.eigh(norm.covariance_matrix)
            cov_mat_xitorch = xitorch.LinearOperator.m(norm.covariance_matrix)
            neigh = torch.linalg.matrix_rank(norm.covariance_matrix).min()
            eigvals_block, eigvectors_block = xitorch.linalg.symeig(cov_mat_xitorch, neig=neigh, mode='uppest')

            locs, probs, ws = get_disc_block(eigvals_block=eigvals_block,
                                         eigvectors_block=eigvectors_block,
                                         mean=norm.nonsparse_loc,
                                         discr_points_per_dim=discr_points_per_dim,
                                         **kwargs)
            self.w = ws

        if hasattr(norm, 'activation'):
            if DEBUG_ACTIVATION:
                locs_act = locs
            else:
                locs_act = norm.activation(locs)
        else:
            locs_act = locs

        super(DiscretizedSparseMultivariateNormal_, self).__init__(probs, locs_act)
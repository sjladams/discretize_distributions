import distributions.test
import wasserstein_distance.test

if __name__ == '__main__':
    ## ---- Distributions ---- ##
    ## -- univariate Distributions -- ##
    ## rectified normal
    # distributions.test.test_rect_norm()
    # distributions.test.test_rect_norm_batches()

    ## mixture normal
    # distributions.test.test_mix_norm()
    # distributions.test.test_mix_norm_batches()

    ## mixture rectified normal
    # distributions.test.test_mix_rect_norm()
    # distributions.test.test_mix_rect_norm_batches()

    ## -- multivariate Distributions -- ##
    ## multivariate normal
    # distributions.test.test_sym_mult_norm()
    # distributions.test.test_mult_norm()
    # distributions.test.test_sym_mult_norm_batches()
    # distributions.test.test_mult_norm_batches()

    ## mixture multivariate normal
    # distributions.test.test_mix_mult_norm()
    # distributions.test.test_mix_mult_norm_batches()

    ## -- Discretization Operations -- ##
    ## Univariate
    # (1) rGMM -> Disc
    # rNorm -> Disc
    # distributions.test.test_rect_norm_to_disc()
    # distributions.test.test_rect_norm_to_disc_batches()

    # # Norm -> Disc
    # distributions.test.test_norm_to_disc()
    # distributions.test.test_norm_to_disc_batches()

    # # GMM -> Disc
    # distributions.test.test_gmm_to_disc()
    # distributions.test.test_gmm_to_disc_batches()

    # rGMM(2) -> Disc
    # distributions.test.test_rgmm_to_disc()
    # distributions.test.test_rgmm_to_disc_batches()

    ## Multivariate
    # Mult Norm -> Disc
    distributions.test.test_mult_norm_to_disc()
    # distributions.test.test_mult_norm_to_disc_batches()

    # Mult Rect Norm -> Disc
    # distributions.test.test_mult_relu_norm_to_disc()
    # distributions.test.test_mult_relu_norm_to_disc_higher_dims()
    # distributions.test.test_mult_relu_norm_to_disc_batches()

    # Sparse Mult Rect Norm -> Disc
    # distributions.test.test_sparse_mult_relu_norm_to_disc()

    ## -- Combining Distributions Operations -- ##
    ## Univariate
    # (2) Disc * Normal -> GMM #
    distributions.test.test_disc_times_normal_to_gmm()
    distributions.test.test_disc_times_normal_to_gmm_batches()

    # distributions.test.test_vec_disc_times_mat_sym_normal_to_gmm()
    # distributions.test.test_vec_disc_times_mat_sym_normal_to_gmm_batches()

    # (2) Disc * GMM -> GMM
    # distributions.test.test_disc_times_gmm_to_gmm()
    # distributions.test.test_disc_times_gmm_to_gmm_batches() # \todo test for different batches

    # (3) GMM + GMM -> GMM
    # distributions.test.test_gmm_plus_gmm()
    # distributions.test.test_gmm_plus_gmm_batches()

    ## Multivariate
    # (1) Mult Disc * Mult Normal -> Mult GMM
    # distributions.test.test_mult_disc_times_mult_normal_to_gmm()
    # distributions.test.test_mult_disc_times_mult_normal_to_gmm_batches()

    # distributions.test.test_vec_disc_times_mat_normal_to_gmm()
    # distributions.test.test_vec_disc_times_mat_normal_to_gmm_batches()

    ## -- Model Reduction Operations - ##
    ## Univariate
    # (4) GMM -> GMM
    # distributions.test.test_simplify_gmm()
    # distributions.test.test_simplify_gmm_batches()

    ## Multivariate
    # distributions.test.test_simplify_mult_gmm()
    # distributions.test.test_simplify_mult_gmm_batches()

    ## -- Expanding Operation -- ##
    # distributions.test.test_expanding_normal()

    ## -- Wasserstein Distance for Operations -- ##
    # (I) rGMM <-W-> Disc
    # rNorm <-W-> Disc
    # wasserstein_distance.test.test_wasserstein_rect_norm_vs_dict()
    # wasserstein_distance.test.test_wasserstein_rect_norm_vs_dict_batches()

    # rGMM(2) <-W-> Disc
    # wasserstein_distance.test.test_wasserstein_rgmm2_vs_dict()
    # wasserstein_distance.test.test_wasserstein_rgmm2_vs_dict_batches()

    # (II) GMM <-W-> GMM
    # wasserstein_distance.test.test_wasserstein_gmm_vs_gmm()
    # wasserstein_distance.test.test_wasserstein_gmm_vs_gmm_batches() # \todo implement

    # (III) Norm <-W-> GMM
    # wasserstein_distance.test.test_wasserstein_norm_vs_gmm()
    # wasserstein_distance.test.test_wasserstein_norm_vs_gmm_batches()
    # wasserstein_distance.test.test_wasserstein_norm_vs_gmm_runs()

    # (IV) Norm <-W-> Norm
    # wasserstein_distance.test.test_wasserstein_norm_vs_norm()
    # wasserstein_distance.test.test_wasserstein_norm_vs_norm_batches()

    # (V) Norm <-W-> rNorm
    # wasserstein_distance.test.test_wasserstein_norm_vs_rect_norm()
    # wasserstein_distance.test.test_wasserstein_norm_vs_rect_norm_batches()

    # (VI) Norm <-W-> trunNorm
    # wasserstein_distance.test.test_wasserstein_norm_vs_trun_norm()
    # wasserstein_distance.test.test_wasserstein_norm_vs_trun_norm_batches()

    # multiNorm <-> MultiNorm
    # wasserstein_distance.test.test_wasserstein_mult_norm_vs_mult_norm()

    ## -- Wasserstein Distance for Networks -- ## #\todo update
    # wasserstein_distance.test.test_wasserstein_fc_batches()

    # wasserstein_distance.test.test_wasserstein_for_different_sizes()



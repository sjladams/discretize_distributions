import pandas as pd
import matplotlib.pyplot as plt

# 23-06-2025
# nr 3 higher dims is basic one
# 4 is with different seed
# 5 is with scaling of cov with 1/sqrt(d)

# df = pd.read_excel("benchmark_results/gmm_2d_discretization_results_nr_components_test_2025-07-03_1.xlsx")
# df = pd.read_excel("benchmark_results/gmm_discretization_results_variance_test_2025-07-07_1.xlsx")
df = pd.read_excel("benchmark_results/gmm_discretization_results_higher_dims_test_2025-07-04_1_no_var_scaling.xlsx")

df_sorted = df.sort_values("num_dims")
plt.figure(figsize=(8, 5))
plt.plot(df_sorted["num_dims"], df_sorted["w2_mix_mean"], label="Multi-Grid", color='blue', alpha=0.6)
plt.plot(df_sorted["num_dims"], df_sorted["w2_old_mean"], label="Per Component", color='red', alpha=0.6)
# plt.plot(df_sorted["num_mix_elems"], (df_sorted["w2_old"]-df_sorted["w2_mix"]), label="W2 diff * 1e1", color='green', alpha=0.6)
plt.xlabel("Dimensions $d$")
plt.ylabel("$\overline{W}_2$")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("benchmark_results/higher_dims_w2_v2.svg")
plt.show()

df_sorted = df.sort_values("num_dims")
plt.figure(figsize=(8, 5))
plt.plot(df_sorted["num_dims"], df_sorted["time_mix_mean"], label="Multi-Grid", color='blue', alpha=0.6)
plt.plot(df_sorted["num_dims"], df_sorted["time_old_mean"], label="Per Component", color='red', alpha=0.6)
# plt.plot(df_sorted["num_mix_elems"], (df_sorted["w2_old"]-df_sorted["w2_mix"]), label="W2 diff * 1e1", color='green', alpha=0.6)
plt.xlabel("Dimensions $d$")
plt.ylabel("Computation time $T$ (sec)")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig("benchmark_results/higher_dims_time_v2.svg")
plt.show()



df_sorted = df.sort_values("num_dims")

plt.figure(figsize=(8, 5))

mean = df_sorted["w2_mix_mean"]
std = df_sorted["w2_mix_std"]
plt.plot(df_sorted["num_dims"], mean, label="Multi-Grid", color='blue', linewidth=2)
plt.plot(df_sorted["num_dims"], mean + std, color='blue', linestyle='--', linewidth=1)
plt.plot(df_sorted["num_dims"], mean - std, color='blue', linestyle='--', linewidth=1)
plt.fill_between(df_sorted["num_dims"], mean - std, mean + std, color='blue', alpha=0.2)

mean = df_sorted["w2_old_mean"]
std = df_sorted["w2_old_std"]
plt.plot(df_sorted["num_dims"], mean, label="Per Component", color='red', linewidth=2)
plt.plot(df_sorted["num_dims"], mean + std, color='red', linestyle='--', linewidth=1)
plt.plot(df_sorted["num_dims"], mean - std, color='red', linestyle='--', linewidth=1)
plt.fill_between(df_sorted["num_dims"], mean - std, mean + std, color='red', alpha=0.2)

plt.xlabel("Dimensions $d$")
plt.ylabel("$\\overline{W}_2$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("benchmark_results/higher_dims_w2_v2.svg")
plt.show()

plt.figure(figsize=(8, 5))

mean = df_sorted["time_mix_mean"]
std = df_sorted["time_mix_std"]
plt.plot(df_sorted["num_dims"], mean, label="Multi-Grid", color='blue', linewidth=2)
plt.plot(df_sorted["num_dims"], mean + std, color='blue', linestyle='--', linewidth=1)
plt.plot(df_sorted["num_dims"], mean - std, color='blue', linestyle='--', linewidth=1)
plt.fill_between(df_sorted["num_dims"], mean - std, mean + std, color='blue', alpha=0.2)

mean = df_sorted["time_old_mean"]
std = df_sorted["time_old_std"]
plt.plot(df_sorted["num_dims"], mean, label="Per Component", color='red', linewidth=2)
plt.plot(df_sorted["num_dims"], mean + std, color='red', linestyle='--', linewidth=1)
plt.plot(df_sorted["num_dims"], mean - std, color='red', linestyle='--', linewidth=1)
plt.fill_between(df_sorted["num_dims"], mean - std, mean + std, color='red', alpha=0.2)

plt.xlabel("Dimensions $d$")
plt.ylabel("Computation time $T$ (sec)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("benchmark_results/higher_dims_time_v2.svg")
plt.show()


# df_sorted = df.sort_values("variance_scale")
# plt.figure(figsize=(8, 5))
# plt.plot(df_sorted["variance_scale"], df_sorted["w2_mix"], label="Multi-Grid", color='blue', alpha=0.6)
# plt.plot(df_sorted["variance_scale"], df_sorted["w2_old"], label="Per Component", color='red', alpha=0.6)
# # plt.plot(df_sorted["num_mix_elems"], (df_sorted["w2_old"]-df_sorted["w2_mix"]), label="W2 diff * 1e1", color='green', alpha=0.6)
# plt.xlabel("Variance scale")
# plt.ylabel("$\overline{W}_2$")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig("benchmark_results/higher_dims_w2_v2.svg")
# plt.show()


# plt.figure(figsize=(8, 5))
# plt.plot(df_sorted["num_dims"], df_sorted["time_mix"], label="W2 Mix",  color='blue', alpha=0.6)
# plt.plot(df_sorted["num_dims"], df_sorted["time_old"], label="W2 Per Component", color='red', alpha=0.6)
# plt.xlabel("Number of Dimensions")
# plt.ylabel("Computation Time (sec)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig("benchmark_results/higher_dims_comp_time.svg")
# plt.show()
#
# df = pd.read_excel("benchmark_results/gmm_discretization_results_test_2025-06-23_1_components.xlsx")
#
# df_sorted = df.sort_values("num_mix_elems")
# plt.figure(figsize=(8, 5))
# plt.plot(df_sorted["num_mix_elems"], df_sorted["w2_mix"], label="W2 Mix", color='blue', alpha=0.6)
# plt.plot(df_sorted["num_mix_elems"], df_sorted["w2_old"], label="W2 Per Component", color='red', alpha=0.6)
# plt.plot(df_sorted["num_mix_elems"], (df_sorted["w2_old"]-df_sorted["w2_mix"]), label="W2 diff",  color='green', alpha=0.6)
# plt.xlabel("Number of Components M")
# plt.ylabel("Normalized Wasserstein-2")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig("benchmark_results/components_w2.svg")
# plt.show()
#
# plt.figure(figsize=(8, 5))
# plt.plot(df_sorted["num_mix_elems"], df_sorted["time_mix"], label="W2 Mix", color='blue', alpha=0.6)
# plt.plot(df_sorted["num_mix_elems"], df_sorted["time_old"], label="W2 Per Component", color='red', alpha=0.6)
# plt.xlabel("Number of Components M")
# plt.ylabel("Computation Time (sec)")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# # plt.savefig("benchmark_results/components_comp_time.svg")
# plt.show()
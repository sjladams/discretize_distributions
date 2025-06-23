import pandas as pd
import matplotlib.pyplot as plt

# 23-06-2025
# nr 3 higher dims is basic one
# 4 is with different seed
# 5 is with scaling of cov with 1/sqrt(d)

df = pd.read_excel("benchmark_results/gmm_discretization_results_test_2025-06-23_5_higher_dim_wth_scaling.xlsx")

df_sorted = df.sort_values("num_dims")
plt.figure(figsize=(8, 5))
plt.plot(df_sorted["num_dims"], df_sorted["w2_mix"], label="W2 Mix", color='blue', alpha=0.6)
plt.plot(df_sorted["num_dims"], df_sorted["w2_old"], label="W2 Per Component", color='red', alpha=0.6)
plt.plot(df_sorted["num_dims"], 10*(df_sorted["w2_old"]-df_sorted["w2_mix"]), label="W2 diff * 1e1", color='green', alpha=0.6)
plt.xlabel("Number of Dimensions")
plt.ylabel("Normalized Wasserstein-2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("benchmark_results/higher_dims_w2.svg")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(df_sorted["num_dims"], df_sorted["time_mix"], label="W2 Mix",  color='blue', alpha=0.6)
plt.plot(df_sorted["num_dims"], df_sorted["time_old"], label="W2 Per Component", color='red', alpha=0.6)
plt.xlabel("Number of Dimensions")
plt.ylabel("Computation Time (sec)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("benchmark_results/higher_dims_comp_time.svg")
plt.show()

df = pd.read_excel("benchmark_results/gmm_discretization_results_test_2025-06-23_1_components.xlsx")

df_sorted = df.sort_values("num_mix_elems")
plt.figure(figsize=(8, 5))
plt.plot(df_sorted["num_mix_elems"], df_sorted["w2_mix"], label="W2 Mix", color='blue', alpha=0.6)
plt.plot(df_sorted["num_mix_elems"], df_sorted["w2_old"], label="W2 Per Component", color='red', alpha=0.6)
plt.plot(df_sorted["num_mix_elems"], (df_sorted["w2_old"]-df_sorted["w2_mix"]), label="W2 diff",  color='green', alpha=0.6)
plt.xlabel("Number of Components M")
plt.ylabel("Normalized Wasserstein-2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("benchmark_results/components_w2.svg")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(df_sorted["num_mix_elems"], df_sorted["time_mix"], label="W2 Mix", color='blue', alpha=0.6)
plt.plot(df_sorted["num_mix_elems"], df_sorted["time_old"], label="W2 Per Component", color='red', alpha=0.6)
plt.xlabel("Number of Components M")
plt.ylabel("Computation Time (sec)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("benchmark_results/components_comp_time.svg")
plt.show()
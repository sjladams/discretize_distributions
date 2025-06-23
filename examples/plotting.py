import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("benchmark_results/gmm_discretization_results_test_2025-06-23_4_higher_dim.xlsx")

df_sorted = df.sort_values("num_dims")
plt.figure(figsize=(8, 5))
plt.plot(df_sorted["num_dims"], df_sorted["w2_mix"], label="W2 Mix", marker='o')
plt.plot(df_sorted["num_dims"], df_sorted["w2_old"], label="W2 Per Component", marker='s')
plt.plot(df_sorted["num_dims"], 10*(df_sorted["w2_old"]-df_sorted["w2_mix"]), label="W2 diff * 1e1")
plt.xlabel("Number of Dimensions")
plt.ylabel("Normalized Wasserstein-2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("benchmark_results/higher_dims_w2.svg")
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(df_sorted["num_dims"], df_sorted["w2_old"]-df_sorted["w2_mix"])
plt.xlabel("Number of Dimensions")
plt.ylabel("Difference Normalized Wasserstein-2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
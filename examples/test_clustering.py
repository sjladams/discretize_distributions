import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Fix points to 2D format
points = np.array([[p] for p in [1, 1.5, 2, 1.3, 4, 2.1]])

# Adjust DBSCAN parameters
eps = 0.5
min_samples = 2

# Fit DBSCAN
clustering = DBSCAN(eps=eps, min_samples=min_samples, algorithm='kd_tree').fit(points)
labels = clustering.labels_

# Plot clusters
unique_labels = set(labels)
colors = ['red', 'blue', 'green', 'purple', 'orange']

for label in unique_labels:
    if label == -1:
        # Noise points
        color = 'black'
        size = 100
    else:
        color = colors[label % len(colors)]
        size = 200

    mask = (labels == label)
    cluster_points = points[mask]

    plt.scatter(cluster_points[:, 0], np.zeros_like(cluster_points[:, 0]), c=color, s=size, label=f'Cluster {label}' if label != -1 else 'Noise')

plt.legend()
plt.xlabel('Value')
plt.yticks([])
plt.title('DBSCAN Clusters')
plt.show()

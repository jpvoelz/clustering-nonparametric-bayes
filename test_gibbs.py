from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
from gibbs_sampler import *
from helper_functions import *

#Choose true parameters for clusters
n_samples=100; centers=4; var=1;iter=100; cluster_std=1; random_state=2

#Generate data
X, y_true = make_blobs(n_samples=n_samples, centers=centers,
                            cluster_std=cluster_std, random_state=random_state, return_centers=False)

#Run Gibbs sampler
y_pred, labels_array, centers, _ = gibbs_sampler(X, var, seed=1, init_cluster_num=10, B=100)

#Calculate Adjusted Rand Index
rand_score = adjusted_rand_score(y_true,y_pred.flatten())

#Plot ground truth
plt.scatter(X[:, 0], X[:, 1], c=y_true, s=40, cmap='Accent');
plt.title("Ground Truth\nN = " + str(n_samples) + "; " + str(centers) + " clusters; variance = " + str(var))
plt.savefig("figures/ground_truth.png")

#Plot predicted labels
plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=40, cmap='Accent');
plt.title("Gibbs Sampler Predicted Labels\nIterations = " + str(iter) + "; Adjusted Rand Index = " + str(round(rand_score,2)))
plt.savefig("figures/predicted_labels.png")

#Make GIF of Gibbs iterations
labels_gif(X, labels_array, "Gibbs Sampler", "figures", "gibbs_sampler")

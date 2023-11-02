# Train neural network using cluster labels as features

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from config import RANDOM_STATE
from datasets import RiceData



from nn_util import nn_trainer
from pca_rice import execute_pca
from ica_rice import execute_ica
from rp_rice import execute_rp
from isomap_rice import execute_isomap


# Load the Rice dataset
# Perform both the clustering and create dataset with these features
print("\nLoading the dataset")
print("- - - - - - - - - - ")
rice_data = RiceData()
X, y = rice_data.get_total()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print("- Rice dataset is loaded")
# Perform KMeans cluster labels as features | Using 2 clusters
kmeans = KMeans(n_clusters=2, random_state=RANDOM_STATE, n_init=10)
kmeans.fit(X)
kmeans_labels = kmeans.predict(X).reshape((-1,1))
KMeans_X = np.concatenate([X, kmeans_labels], axis=1)
kmeans_x_train, kmeans_x_test, y_train, y_test = train_test_split(KMeans_X, y, test_size=0.2, random_state=RANDOM_STATE)
print("- Rice dataset with KMeans labels is loaded")
# Perform EM clustering as features | Using 2 clusters
em = GaussianMixture(n_components=2, random_state=RANDOM_STATE)
em.fit(X)
em_labels = kmeans.predict(X).reshape((-1,1))
EM_X = np.concatenate([X, em_labels], axis=1)
em_x_train, em_x_test, y_train, y_test = train_test_split(EM_X, y, test_size=0.2, random_state=RANDOM_STATE)
print("- Rice dataset with EM labels is loaded")

# Start figures to plot the training and testing curves
train_fig = plt.figure(figsize=(8,6))
test_fig = plt.figure(figsize=(8,6))
train_ax = train_fig.add_subplot(1,1,1)
test_ax = test_fig.add_subplot(1,1,1)

print("\nTraining network")
print("- - - - - - - - -")
# Train network on Rice data
print("Begin training on Rice dataset")
iterations, training_score, testing_score, _, training_time = nn_trainer(
    x_train, y_train, x_test, y_test, result_dir="./results/nn_cluster/rice"
    )
train_ax.plot(iterations, training_score, color="black", label="Rice data")
test_ax.plot(iterations, testing_score, color="black", label="Rice data")
vanilla_training_time = training_time
# Train network on Rice data with KMeans labels
print("Begin training on Rice dataset with KMeans labels")
iterations, training_score, testing_score, _, training_time = nn_trainer(
    kmeans_x_train, y_train, kmeans_x_test, y_test, result_dir="./results/nn_cluster/kmeans_rice"
    )
train_ax.plot(iterations, training_score, linestyle=":", color="blue", label="Rice data with KMeans clustering")
test_ax.plot(iterations, testing_score, linestyle=":", color="blue", label="Rice data with KMeans clustering")
kmeans_training_time = training_time
# Train network on Rice data with EM labels
print("Begin training on Rice dataset with EM labels")
iterations, training_score, testing_score, _, training_time = nn_trainer(
    em_x_train, y_train, em_x_test, y_test, result_dir="./results/nn_cluster/em_rice"
    )
train_ax.plot(iterations, training_score, linestyle="--", color="green", label="Rice data with EM clustering")
test_ax.plot(iterations, testing_score, linestyle="--", color="green", label="Rice data with EM clustering")
em_training_time = training_time

# Set titles, labels, and legends to the plots
train_ax.set_xlabel("Iterations")
train_ax.set_ylabel("F1 score")
train_ax.set_title("Training F1-score VS iterations")
train_ax.legend()
test_ax.set_xlabel("Iterations")
test_ax.set_ylabel("F1 score")
test_ax.set_title("Testing F1-score VS iterations")
test_ax.legend()
# Save the plots
train_fig.savefig("./plots/cluster_nn_train.png")
test_fig.savefig("./plots/cluster_nn_test.png")
print("Plot of training curves saved at: './plots/cluster_nn_train.png")
print("Plot of testing curves saved at: './plots/cluster_nn_test.png")

# # print the training times
# print("\nTraining times")
# print("- - - - - - - -")
# print(f"- Training time for Rice dataset: {vanilla_training_time:.2f} s")
# print(f"- Training time for Rice dataset reduced by PCA: {pca_training_time:.2f} s")
# print(f"- Training time for Rice dataset reduced by ICA: {ica_training_time:.2f} s")
# print()



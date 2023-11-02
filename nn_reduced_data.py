# Train neural network for reduced dataset

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE
from datasets import RiceData
from nn_util import nn_trainer
from pca_rice import execute_pca
from ica_rice import execute_ica
from rp_rice import execute_rp
from isomap_rice import execute_isomap

# Load the Rice dataset
# Reduce using all dim. reduction algorithm
# Also split each dataset into drain test part
print("\nLoading the dataset")
print("- - - - - - - - - -")
rice_data = RiceData()
X, y = rice_data.get_total()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print("- Rice dataset is loaded")
# - PCA: 0.99 variance as the optimum number of compnents
PCA_X, _, pc_cnt = execute_pca(X)
PCA_X = PCA_X[:, 0:pc_cnt]
pca_x_train, pca_x_test, y_train, y_test = train_test_split(PCA_X, y, test_size=0.2, random_state=RANDOM_STATE)
print("- PCA reduced Rice dataset is loaded")
# - ICA: 6 as the optimum number of components
ICA_X, _ = execute_ica(X, n_comp=6)
ica_x_train, ica_x_test, y_train, y_test = train_test_split(ICA_X, y, test_size=0.2, random_state=RANDOM_STATE)
print("- ICA reduced Rice dataset is loaded")
# - RP: 4 as optimum number of components
RP_X, _ = execute_rp(X, n_comp=6)
rp_x_train, rp_x_test, y_train, y_test = train_test_split(RP_X, y, test_size=0.2, random_state=RANDOM_STATE)
print("- RP reduced Rice dataset is loaded")
# - Isomap: 3 as optimum number of components
Isomap_X, _ = execute_isomap(X, n_comp=3)
isomap_x_train, isomap_x_test, y_train, y_test = train_test_split(Isomap_X, y, test_size=0.2, random_state=RANDOM_STATE)
print("- Isomap reduced Rice dataset is loaded")

# Start figures to plot the training and testing curves
train_fig = plt.figure(figsize=(8,6))
test_fig = plt.figure(figsize=(8,6))
train_ax = train_fig.add_subplot(1,1,1)
test_ax = test_fig.add_subplot(1,1,1)

print("\nTraining network")
print("- - - - - - - - -")
# Train network on vanilla Rice data
print("Begin training on Rice dataset")
iterations, training_score, testing_score, _, training_time = nn_trainer(
    x_train, y_train, x_test, y_test, result_dir="./results/nn_reduced/rice"
    )
train_ax.plot(iterations, training_score, color="black", label="Rice data")
test_ax.plot(iterations, testing_score, color="black", label="Rice data")
vanilla_training_time = training_time
# Train network on PCA reduced Rice data
print("Begin training on Rice dataset reduced by PCA")
iterations, training_score, testing_score, _, training_time = nn_trainer(
    pca_x_train, y_train, pca_x_test, y_test, result_dir="./results/nn_reduced/pca_rice"
    )
train_ax.plot(iterations, training_score, color="blue", label="PCA Rice data")
test_ax.plot(iterations, testing_score, color="blue", label="PCA Rice data")
pca_training_time = training_time
# Train network on ICA reduced Rice data
print("Begin training on Rice dataset reduced by ICA")
iterations, training_score, testing_score, _, training_time = nn_trainer(
    ica_x_train, y_train, ica_x_test, y_test, result_dir="./results/nn_reduced/ica_rice"
    )
train_ax.plot(iterations, training_score, color="green", label="ICA Rice data")
test_ax.plot(iterations, testing_score, color="green", label="ICA Rice data")
ica_training_time = training_time
# Train network on RP reduced Rice data
print("Begin training on Rice dataset reduced by RP")
iterations, training_score, testing_score, _, training_time = nn_trainer(
    rp_x_train, y_train, rp_x_test, y_test, result_dir="./results/nn_reduced/rp_rice"
    )
train_ax.plot(iterations, training_score, color="red", label="RP Rice data")
test_ax.plot(iterations, testing_score, color="red", label="RP Rice data")
rp_training_time = training_time
# Train network on Isomap reduced Rice data
print("Begin training on Rice dataset reduced by Isomap")
iterations, training_score, testing_score, _, training_time = nn_trainer(
    isomap_x_train, y_train, isomap_x_test, y_test, result_dir="./results/nn_reduced/isomap_rice"
    )
train_ax.plot(iterations, training_score, color="orange", label="Isomap Rice data")
test_ax.plot(iterations, testing_score, color="orange", label="Isomap Rice data")
isomap_training_time = training_time

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
train_fig.savefig("./plots/reduced_nn_train.png")
test_fig.savefig("./plots/reduced_nn_test.png")
print("Plot of training curves saved at: './plots/reduced_nn_train.png")
print("Plot of testing curves saved at: './plots/reduced_nn_test.png")

# print the training times
print("\nTraining times")
print("- - - - - - - -")
print(f"- Training time for Rice dataset: {vanilla_training_time:.2f} s")
print(f"- Training time for Rice dataset reduced by PCA: {pca_training_time:.2f} s")
print(f"- Training time for Rice dataset reduced by ICA: {ica_training_time:.2f} s")
print(f"- Training time for Rice dataset reduced by Random Projection: {rp_training_time:.2f} s")
print(f"- Training time for Rice dataset reduced by Isomap: {isomap_training_time:.2f} s")
print()

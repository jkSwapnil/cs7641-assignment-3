

"""
Main function for neural network training using Stochastic Gradient Descent
"""

import json
import os
import time

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm

from config import RANDOM_STATE

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def nn_trainer(x_train, y_train, x_test, y_test, result_dir):
    """ Function to implement neural network training using Stochastic Gradient Descent
    Parameters:
        x_train: Input training features (np.ndarray)
        y_train: Training labels (np.ndarray)
        x_test: Input testing features (np.ndarray)
        y_test: Input testing features (np.ndarray)
        result_dir: Path to directory to save results in (str)
    Returns:
        iterations: Training iterations
        training_score: Training score for training curves
        testing_score: Testing score for testing curve
        test_results: Test results
        training_time: Time of full training
    """
    # Train and test the model for different number of iterations
    iterations = []
    training_score = []
    testing_score = []
    test_results = {}  # Test results for the best weights
    training_time = 0  # Time taken to train the neural network
    for i in tqdm(range(1, 105, 4)):
        # Set the max iterations
        if i == 0:
            max_iters = 1
            iterations.append(max_iters)
        else:
            max_iters = i
            iterations.append(max_iters)
        # Define the model
        model = MLPClassifier(
            hidden_layer_sizes=[100], activation="logistic", solver="sgd", alpha=0,
            learning_rate_init=0.001, max_iter=max_iters, random_state=RANDOM_STATE
            )
        # Train the neural network
        # Use F1 score for train curve
        start_time = time.time()
        model.fit(x_train, y_train)
        training_time = time.time() - start_time  # Training time
        y_train_pred = model.predict(x_train)
        f1_train = f1_score(y_train, y_train_pred)  # Predict training score
        training_score.append(f1_train)
        # Generate the testing metrics: % accuracy, f1-score, and roc_score
        # Use F1 score for test curve
        y_test_pred = model.predict(x_test)
        y_test_pred_proba = model.predict(x_test)
        percentage_accuracy = accuracy_score(y_test, y_test_pred)*100
        f1_test = f1_score(y_test, y_test_pred)
        roc_auc = roc_auc_score(y_test, y_test_pred_proba)
        testing_score.append(f1_test)
        if not test_results or f1_test >= test_results["f1_score"]:
            test_results["accuracy"] = percentage_accuracy
            test_results["f1_score"] = f1_test
            test_results["roc_auc"] = roc_auc
    # Save the results
    with open(os.path.join(result_dir, "test_result.json"), "w") as f:
        json.dump(test_results, f)
    print("\t- Test results saved at: ", os.path.join(result_dir, "test_result.json"))

    return iterations, training_score, testing_score, test_results, training_time


if __name__ == "__main__":

    from datasets import RiceData

    # Load the digits dataset
    # Get the train and test part and train the model
    rice_data = RiceData()
    x_train, y_train = rice_data.get_train()
    x_test, y_test = rice_data.get_test()
    iterations, training_score, testing_score, test_results, training_time = nn_trainer(
        x_train, y_train, x_test, y_test, result_dir="./results/example"
        )

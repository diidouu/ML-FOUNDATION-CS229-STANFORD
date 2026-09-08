import matplotlib.pyplot as plt
import numpy as np
import util
import os

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    best_tau = None
    best_mse = float('inf')
    
    for tau in tau_values:
        model = LocallyWeightedLinearRegression(tau)
        model.fit(x_train, y_train)
        preds = model.predict(x_valid)
        mse = np.mean((preds - y_valid) ** 2)
        
        if mse < best_mse:
            best_mse = mse
            best_tau = tau
    
    # Fit a LWR model with the best tau value
    model = LocallyWeightedLinearRegression(best_tau)
    model.fit(x_train, y_train)
    # Run on the test set to get the MSE value
    preds = model.predict(x_test)
    test_mse = np.mean((preds - y_test) ** 2)
    
    # Save predictions to pred_path
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    np.savetxt(pred_path, preds)
    
    # Plot data
    plt.figure()
    plt.plot(x_train[:, 1], y_train, 'bx', label='train')
    plt.plot(x_test[:, 1], y_test, 'go', label='test')
    plt.plot(x_test[:, 1], preds, 'ro', label='predicted')
    plt.title(f'LWR with tau={best_tau}, Test MSE={test_mse:.4f}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()    
    # *** END CODE HERE ***

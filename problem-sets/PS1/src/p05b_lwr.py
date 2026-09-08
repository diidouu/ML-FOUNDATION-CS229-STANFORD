import matplotlib.pyplot as plt
import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)
    preds = model.predict(x_eval)
    # Get MSE value on the validation set
    mse = np.mean((preds - y_eval) ** 2)
    # Plot validation predictions on top of training set
    plt.scatter(x_train[:, 1], y_train, label='Training data')
    plt.scatter(x_eval[:, 1], preds, color='red', label='LWR Predictions')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'LWR Predictions (tau={tau}), MSE={mse:.4f}')
    plt.legend()
    plt.show()
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x 
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m,n = x.shape
        preds = np.zeros(m)
        for i in range(m):
            diff = self.x - x[i]
            w = np.exp(-np.sum(diff**2 , axis=1) / (2 * self.tau ** 2))
            W = np.diag(w)
            theta = np.linalg.solve(self.x.T @ W @ self.x, self.x.T @ W @ self.y)
            preds[i] = x[i] @ theta
        return preds
        # *** END CODE HERE ***

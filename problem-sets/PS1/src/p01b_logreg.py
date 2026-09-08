import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)

    # Plot training data
    plt.figure()
    plt.scatter(x_train[y_train == 0, 1], x_train[y_train == 0, 2], marker='o', label='y=0')
    plt.scatter(x_train[y_train == 1, 1], x_train[y_train == 1, 2], marker='x', label='y=1')
    plt.legend()
    plt.title("Training data")

    # *** START CODE HERE ***
    model = LogisticRegression()
    model.fit(x_train, y_train)

    # Plot decision boundary
    theta = model.theta
    x1_vals = np.linspace(np.min(x_train[:, 1]), np.max(x_train[:, 1]), 100)
    x2_vals = -(theta[0] + theta[1] * x1_vals) / theta[2]
    plt.plot(x1_vals, x2_vals, label='Decision boundary')
    plt.legend()
    plt.title("LOGISTIC REGRESSION : Decision boundary after training")
    plt.savefig('output/logreg_decision_boundary.png', dpi=300)

    plt.show()
    

    preds = model.predict(x_eval)
    np.savetxt(pred_path, preds)
    # *** END CODE HERE ***



class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n)
        
        for iter in range(self.max_iter):
            h = 1 / (1 + np.exp(-(x @ self.theta)))
            cost_function = -1/m * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h))
            gradient = -1/m * x.T @ (y - h)
            R = np.diag(h * (1 - h))
            H = (1/m) * x.T @ R @ x
            delta = np.linalg.solve(H, gradient)
            theta_updated = self.theta - delta
            if np.linalg.norm(theta_updated - self.theta) < self.eps:
                break
            self.theta = theta_updated
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        z = x @ self.theta
        prediction = 1 / (1 + np.exp(-z))
        return prediction
        # *** END CODE HERE ***

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
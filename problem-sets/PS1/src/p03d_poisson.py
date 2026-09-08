import numpy as np
import util

from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_eval, _ = util.load_dataset(eval_path, add_intercept=True)
    
    # The line below is the original one from Stanford. It does not include the intercept, but this should be added.
    # x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    model = PoissonRegression(step_size=lr)
    model.fit(x_train, y_train)
    
    preds = model.predict(x_eval)
    np.savetxt(pred_path, preds)
    
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    # *** END CODE HERE ***


class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        self.theta = np.zeros(n)
        
        for _ in range(self.max_iter):
            # Compute predictions with clipped logits to prevent overflow in exp()
            z = x @ self.theta
            z = np.clip(z, -500, 500)  # Prevent exp overflow/underflow (exp(500) ≈ inf, exp(-500) ≈ 0)
            eta = np.exp(z)
            
            # Gradient ascent step: maximize log-likelihood
            gradient = x.T @ (y - eta)
            
            # Adaptive step scaling to prevent theta divergence with clipping
            gradient_norm = np.linalg.norm(gradient)
            if gradient_norm > 1.0:
                gradient = gradient / (gradient_norm + 1e-8)  # Normalize to unit norm if too large
            
            self.theta += self.step_size * gradient
            
            # Optional: Clip theta to prevent unbounded growth
            self.theta = np.clip(self.theta, -100, 100)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        z = x @ self.theta
        z = np.clip(z, -500, 500)  # Prevent overflow in exp; exp(500) ≈ inf, exp(-500) ≈ 0
        return np.exp(z)
        # *** END CODE HERE ***
        
if __name__ == "__main__":
    import sys
    main(
        float(sys.argv[1]),
        sys.argv[2],
        sys.argv[3],
        sys.argv[4]
    )
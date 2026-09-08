import numpy as np
import util
import matplotlib.pyplot as plt

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)"""
    # Charger les datasets
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)

    # Entraîner le modèle
    model = GDA()
    model.fit(x_train, y_train)

    # Faire des prédictions
    preds = model.predict(x_eval)
    np.savetxt(pred_path, preds)  # Sauvegarder les prédictions

    # --- Visualisation ---
    # On ne prend que les 2 premières dimensions pour le graphe
    X_plot = x_eval[:, :2]
    y_plot = y_eval

    # Affichage des points réels
    plt.scatter(X_plot[y_plot==0, 0], X_plot[y_plot==0, 1], color='blue', label='Classe 0')
    plt.scatter(X_plot[y_plot==1, 0], X_plot[y_plot==1, 1], color='red', label='Classe 1')

    # Points mal classés
    misclassified = preds != y_plot
    plt.scatter(X_plot[misclassified, 0], X_plot[misclassified, 1], 
                facecolors='none', edgecolors='black', s=100, label='Mal classé')

    # Frontière de décision
    sigma_inv = np.linalg.inv(model.sigma)
    theta = sigma_inv @ (model.mu1 - model.mu0)
    theta0 = (
        -0.5 * model.mu1.T @ sigma_inv @ model.mu1
        + 0.5 * model.mu0.T @ sigma_inv @ model.mu0
        + np.log(model.phi / (1 - model.phi))
    )

    # Grille pour tracer la frontière
    x_vals = np.linspace(np.min(X_plot[:,0])-0.5, np.max(X_plot[:,0])+0.5, 200)
    y_vals = np.linspace(np.min(X_plot[:,1])-0.5, np.max(X_plot[:,1])+0.5, 200)
    xx, yy = np.meshgrid(x_vals, y_vals)
    grid = np.c_[xx.ravel(), yy.ravel()]
    scores = grid @ theta + theta0
    zz = scores.reshape(xx.shape)

    # Tracer la frontière de décision
    plt.contour(xx, yy, zz, levels=[0], colors='green', linewidths=2, linestyles='--')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('GDA : Points réels, prédictions et frontière de décision')
    plt.legend()
    plt.savefig('output/gda_decision_boundary.png', dpi=300)

    plt.show()
    
    
    


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        
        self.phi = np.mean(y)
        self.mu0 = np.mean(x[y == 0], axis=0)
        self.mu1 = np.mean(x[y == 1], axis=0)
        self.sigma = np.zeros((n, n))
        
        for i in range(m):
            if y[i] == 0:
                diff = (x[i] -self.mu0).reshape(-1, 1)
            else:
                diff = (x[i] - self.mu1).reshape(-1, 1)
            self.sigma += diff @ diff.T
        self.sigma /= m
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        sigma_inv = np.linalg.inv(self.sigma)
        theta = sigma_inv @ (self.mu1 - self.mu0)
        theta0 = (
            0.5 * (self.mu0 + self.mu1).T @ sigma_inv @ (self.mu0 - self.mu1) 
            - np.log((1 - self.phi) / self.phi)
        )
        
        score = x @ theta + theta0
        prediction = (score >= 0).astype(int)
        return prediction
        # *** END CODE HERE*
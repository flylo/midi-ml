import numpy as np


class PrincipalComponents(object):
    """
    Run a principal components analysis
    """

    def __init__(self, X, regularization: float = None):
        self.X = X.astype(float)
        self.means_ = None  # type: np.array
        self.sds_ = None  # type: np.array
        self.covariance_ = None  # type: np.array
        self.eigenvalues_ = None  # type: np.array
        self.projection_matrix_ = None  # type: np.array
        self.num_records_ = self.X.shape[0]
        self.regularization_ = regularization

    def _normalize(self, new_X: np.array = None) -> np.array:
        """
        Subtract the mean and divide by the standard deviation.
        If you do not subtract the mean from your data, the principal components will project in the direction of the
        largest features by magnitude.
        :param new_X:
        :return:
        """
        if new_X is None:
            self.means_ = self.X.mean(axis=0)
            self.sds_ = np.sqrt(np.power(self.X - self.means_, 2).mean(axis=0))
            return (self.X - self.means_) / self.sds_
        else:
            return (new_X - self.means_) / self.sds_

    def _get_covariance(self):
        """

        :return:
        """
        self.X = self._normalize()
        self.covariance_ = np.dot(self.X.T, self.X) / (self.num_records_ - 1)
        if self.regularization_ is not None:
            reg_covariance = self.regularization_ * self.covariance_
            reg_covariance += (1 - self.regularization_) * np.eye(self.covariance_.shape[1])
            self.covariance_ = reg_covariance

    def _eigendecomposition(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.covariance_)
        diagonal_eigenvalues = np.eye(self.X.shape[1]) * eigenvalues.T
        eigendecomposition = np.linalg.solve(eigenvectors.T, eigenvectors.T.dot(diagonal_eigenvalues))
        eigenvalue_order = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[eigenvalue_order]
        self.projection_matrix_ = eigendecomposition[:, eigenvalue_order]

    def transform(self, new_X: np.array = None) -> np.array:
        if self.projection_matrix_ is None:
            raise ValueError("Must fit model before transformation")
        if new_X is None:
            return self.X.dot(self.projection_matrix_)
        else:
            return self._normalize(new_X).dot(self.projection_matrix_)

    def fit(self):
        self._normalize()
        self._get_covariance()
        self._eigendecomposition()


def main():
    X = np.array([[1, 2, 3],
                  [2, 1, 2],
                  [5, 4, 6],
                  [1, 2, 9]])
    pca = PrincipalComponents(X)
    pca.fit()


if __name__ == "__main__":
    main()

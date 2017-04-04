import numpy as np


# TODO: docstrings
class PrincipalComponents(object):
    """
    Run a principal components analysis
    """
    _methods = ["eigendecomposition", "svd"]

    def __init__(self, X,
                 method="eigendecomposition",
                 regularization: float = None,
                 num_components: float = None,
                 keep_copy_of_X: bool = True, ):
        """
        :param X: Input data
        :param regularization: (optional) whether we want to regularize the covariance matrix to avoid singularity
        :param num_components: (optional) the number of PCs to keep
        :param keep_copy_of_X: whether to keep a copy of the training data once we train
        """
        self.X = X.astype(float)
        if method in self._methods:
            self.method_ = method
        else:
            raise ValueError("Must choose method in %s" % self._methods)
        self.means_ = None  # type: np.array
        self.sds_ = None  # type: np.array
        self.covariance_ = None  # type: np.array
        self.eigenvalues_ = None  # type: np.array
        self.projection_matrix_ = None  # type: np.array
        self.num_components_ = num_components
        self.num_records_ = self.X.shape[0]
        self.regularization_ = regularization
        self.keep_copy_of_X = keep_copy_of_X

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
            # set sds to 1 if there is no variance so normalization doesn't return NaN
            self.sds_[self.sds_ == 0] = 1
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
        """

        :return:
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.covariance_)
        diagonal_eigenvalues = np.eye(self.X.shape[1]) * eigenvalues.T
        eigendecomposition = np.linalg.solve(eigenvectors.T, eigenvectors.T.dot(diagonal_eigenvalues))
        eigenvalue_order = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[eigenvalue_order]
        self.projection_matrix_ = eigendecomposition[:, eigenvalue_order]

    def _svd(self):
        """

        :return:
        """
        u, s, v = np.linalg.svd(self.X, full_matrices=False)
        self.eigenvalues_ = s**2 / self.X.shape[0]
        # self.projection_matrix_ = v.T
        self.projection_matrix_ = v.T

    def transform(self, new_X: np.array = None) -> np.array:
        if self.projection_matrix_ is None:
            raise ValueError("Must fit model before transformation")
        if new_X is None:
            if self.keep_copy_of_X:
                return self.X.dot(self.projection_matrix_)
            else:
                raise ValueError("Must keep set keep_copy_of_X to True")
        else:
            return self._normalize(new_X).dot(self.projection_matrix_)

    def fit(self):
        self._normalize()
        if self.method_ == "eigendecomposition":
            self._get_covariance()
            self._eigendecomposition()
        else:
            self._svd()
        if self.num_components_ is not None:
            self.eigenvalues_ = self.eigenvalues_[:self.num_components_]
            self.projection_matrix_ = self.projection_matrix_[:,:self.num_components_]
        if not self.keep_copy_of_X:
            self.X = None


def main():
    X = np.array([[1, 2, 3],
                  [2, 1, 2],
                  [5, 4, 6],
                  [1, 2, 9]])
    pca = PrincipalComponents(X)
    pca.fit()


if __name__ == "__main__":
    main()

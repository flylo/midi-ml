import numpy as np
from midi_ml.models.decomposition import PrincipalComponents

class SubspaceClassifier(object):
    def __init__(self,
                 X: np.array,
                 y: np.array,
                 explained_variance: float = 0.9,
                 covariance_regularization: float = None,
                 keep_copy_of_X: bool = True):
        self.X = X
        self.y = y
        self.classes_ = set(self.y)
        self.means_ = None  # type: np.array
        self.sds_ = None  # type: np.array
        self.explained_variance_ = explained_variance
        self.covariance_regularization_ = covariance_regularization
        self.keep_copy_of_X = keep_copy_of_X
        self.class_subspace_projection_operators_ = {}

    # TODO: make decomposition and subspace classifiers each inherit from the same base class
    def _normalize(self, new_X: np.array = None) -> np.array:
        """
        Subtract the mean and divide by the standard deviation.
        :param new_X:
        :return:
        """
        if new_X is None:
            self.means_ = self.X.mean(axis=0)
            self.sds_ = np.sqrt(np.power(self.X - self.means_, 2).mean(axis=0))
            return (self.X - self.means_) / self.sds_
        else:
            return (new_X - self.means_) / self.sds_

    def predict(self, new_X: np.array = None) -> np.array:
        """
        Exposed API to make predictions
        :param new_X: New set of X values to make predictions with (optional)
        :return:
        """
        if self.class_subspace_projection_operators_ is None:
            raise ValueError("Must fit model before transformation")
        projection_distance_to_class = {}
        for c in self.classes_:
            subspace_projection_operator = self.class_subspace_projection_operators_[c]  # type: np.array
            if new_X is None:
                if self.keep_copy_of_X:
                    if self.X.shape[1] > subspace_projection_operator.shape[1]:
                        padding = np.zeros((subspace_projection_operator.shape[0],
                                            self.X.shape[1] - subspace_projection_operator.shape[1]))
                        subspace_projection_operator = np.hstack([subspace_projection_operator, padding])
                    projection_distance_to_class[c] = np.power(self.X.dot(subspace_projection_operator), 2).sum(axis=1)
                else:
                    raise ValueError("Must keep set keep_copy_of_X to True")
            else:
                standardized_X = self._normalize(self.X)
                projection_distance_to_class[c] = np.power(standardized_X.dot(subspace_projection_operator), 2) \
                    .sum(axis=1)

        distances = np.vstack([projection_distance_to_class[i] for i in self.classes_]).T
        return distances.argmax(axis=1)

    def fit(self):
        """
        Learn class conditional principal components
        :return:
        """
        self.X = self._normalize()
        for c in self.classes_:
            X_c = self.X[np.where(self.y == c)]
            pc = PrincipalComponents(X_c,
                                     regularization=self.covariance_regularization_,
                                     keep_copy_of_X=self.keep_copy_of_X)
            pc.fit()
            percentage_variance = (pc.eigenvalues_ / pc.eigenvalues_.sum()).cumsum()
            idx = np.where(percentage_variance >= self.explained_variance_)[0][0]
            subspace_projection_operator = pc.projection_matrix_[:, :idx + 1]
            self.class_subspace_projection_operators_[c] = subspace_projection_operator
        if not self.keep_copy_of_X:
            self.X = None


def main():
    from sklearn import datasets
    X, y = datasets.make_classification(n_samples=1000,
                                        n_features=2,
                                        n_informative=2,
                                        n_redundant=0,
                                        random_state=1010)
    # X, y = datasets.make_classification(n_samples=1000,
    #                                     n_features=100,
    #                                     n_informative=2,
    #                                     n_redundant=98,
    #                                     random_state=1010)
    sc = SubspaceClassifier(X, y, covariance_regularization=0.3)
    sc.fit()
    sc.predict()
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y, sc.predict()))
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()

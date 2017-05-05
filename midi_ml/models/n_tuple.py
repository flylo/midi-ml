import numpy as np

import pdb


# TODO: invert the dict so that we search by pattern before class
class NTupleClassifier(object):
    """
    """

    def __init__(self,
                 X: np.array,
                 y: np.array,
                 M: int = 10,
                 L: int = 5,
                 num_bins=20,
                 keep_copy_of_X=True):
        """
        :param X: (N * M)-dimensional array containing the input data in matrix form
        :param y: (N * 1)-dimensional array containing the binary target variable, encoded as 0 and 1
        :param keep_copy_of_X: whether or not to persist a copy of the data in the model object
        """
        self.X = X
        self.y = y
        self.M = M
        self.L = L
        self.T_mk = {}
        # must be integers starting at 0 and increasing by 1
        self.classes_ = set(self.y)
        self.keep_copy_of_X = keep_copy_of_X
        self.num_bins = num_bins
        self.X_discretized = None  # type: np.array
        self.pattern_sets = {}
        self.feature_bins = {}
        self.fitted = False

    def _predict(self, X) -> np.array:
        pass

    def predict(self, new_X: np.array = None):
        """
        Exposed API to make predictions
        :param new_X: New set of X values to make predictions with (optional)
        :return:
        """
        if new_X is None:
            if not self.keep_copy_of_X:
                raise ValueError("Must keep copy of X in order to make predictions")
            if not self.fitted:
                raise LookupError("Model must be trained in order to make predicitons")
            T_mk_b = {k: np.zeros((self.X.shape[0], self.M)) for k in self.classes_}
            for i, x in enumerate(self.X_discretized):
                for m in range(self.M):
                    binary = ""
                    for l in range(self.L):
                        locs = self.pattern_sets[m][l]
                        # since each feature is a discretized continuous RV, we
                        # are checking if the random index is equal to the actual
                        # value of the categorical (discretized RV) feature
                        if x[locs[0]] == locs[1]:
                            binary += str(1)
                        else:
                            binary += str(0)
                    for cls in self.classes_:
                        try:
                            T_mk_b[cls][i, m] = self.T_mk[cls][m][binary]
                        except KeyError:
                            # if there were no instances of this pattern
                            # in the training set, then we keep the prediction
                            # as probability 0.0
                            continue
            # we use the decision rule f_k = sum_{m=1}^M T_mk(b_m)
            class_T_mk_sum = [T_mk_b[cls].sum(axis=1).reshape((self.X.shape[0], 1)) for cls in self.classes_]
            class_T_mk_sum = np.concatenate(class_T_mk_sum, axis=1)
            f_k = class_T_mk_sum.argmax(axis=1)
            return f_k

        else:
            new_X_discretized = self._discretize_prediction_data(new_X)
            T_mk_b = {k: np.zeros((new_X.shape[0], self.M)) for k in self.classes_}
            for i, x in enumerate(new_X_discretized):
                for m in range(self.M):
                    binary = ""
                    for l in range(self.L):
                        locs = self.pattern_sets[m][l]
                        # since each feature is a discretized continuous RV, we
                        # are checking if the random index is equal to the actual
                        # value of the categorical (discretized RV) feature
                        if x[locs[0]] == locs[1]:
                            binary += str(1)
                        else:
                            binary += str(0)
                    for cls in self.classes_:
                        try:
                            T_mk_b[cls][i, m] = self.T_mk[cls][m][binary]
                        except KeyError:
                            # if there were no instances of this pattern
                            # in the training set, then we keep the prediction
                            # as probability 0.0
                            continue
            # we use the decision rule f_k = sum_{m=1}^M T_mk(b_m)
            class_T_mk_sum = [T_mk_b[cls].sum(axis=1).reshape((new_X.shape[0], 1)) for cls in self.classes_]
            class_T_mk_sum = np.concatenate(class_T_mk_sum, axis=1)
            # f_k = class_T_mk_sum.argmax(axis=1)
            f_k = class_T_mk_sum
            return f_k

    def _discretize_prediction_data(self, new_X: np.array = None) -> np.array:
        X_discretized = np.zeros(new_X.shape)
        # don't count columns that have 0 variance
        for feature, bins in self.feature_bins.items():
            if bins is 0:
                continue
            else:
                X_bins = np.histogram(new_X[:, feature], bins=bins)[1]
                X_discretized[:, feature] = np.digitize(new_X[:, feature], X_bins)
        return X_discretized

    def _discretize_training_data(self):
        X_discretized = np.zeros(self.X.shape)
        # don't count columns that have 0 variance
        for k, v in enumerate(self.X.sum(axis=0)):
            if v is 0:
                self.feature_bins[k] = 0
            else:
                X_bins = np.histogram(self.X[:, k], bins=self.num_bins)[1]
                self.feature_bins[k] = X_bins
                X_discretized[:, k] = np.digitize(self.X[:, k], X_bins)
        self.X_discretized = X_discretized

    def fit(self):
        """
        Exposed API for training an n-tuple classifier
        :return:
        """
        self._discretize_training_data()

        # create pattern sets
        possible_columns = np.arange(self.X.shape[1])
        possible_records = np.arange(self.num_bins)
        for m in range(self.M):
            self.pattern_sets[m] = {}
            for l in range(self.L):
                random_column = np.random.choice(possible_columns)
                random_record = np.random.choice(possible_records)
                self.pattern_sets[m][l] = (random_column, random_record)

        for cls in self.classes_:
            self.T_mk[cls] = {m: {} for m in range(self.M)}
            cls_idx = np.where(self.y == cls)[0]
            cls_size = cls_idx.shape[0]
            for row in cls_idx:
                x = self.X_discretized[row, :]
                for m in range(self.M):
                    binary = ""
                    for l in range(self.L):
                        locs = self.pattern_sets[m][l]
                        # since each feature is a discretized continuous RV, we
                        # are checking if the random index is equal to the actual
                        # value of the categorical (discretized RV) feature
                        if x[locs[0]] == locs[1]:
                            binary += str(1)
                        else:
                            binary += str(0)
                    try:
                        self.T_mk[cls][m][binary] += 1. / (cls_size)
                    except KeyError:
                        self.T_mk[cls][m][binary] = 1. / (cls_size)
        self.fitted = True


if __name__ == "__main__":
    import numpy as np
    from sklearn import datasets, metrics, cross_validation
    from midi_ml.models.n_tuple import *

    X, y = datasets.make_classification(n_samples=1000,
                                        n_features=50,
                                        n_informative=40,
                                        n_redundant=10,
                                        random_state=1010)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)
    nt = NTupleClassifier(X_train, y_train, M=200, L=5, num_bins=5)
    nt.fit()
    preds = nt.predict(new_X=X_test).argmax(axis=1)
    metrics.confusion_matrix(y_test, preds)
    metrics.confusion_matrix(y_train, nt.predict())

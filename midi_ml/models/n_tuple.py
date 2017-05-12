import numpy as np
import pdb


class NTupleClassifier(object):
    """
    """

    def __init__(self,
                 X: np.array,
                 y: np.array,
                 M: int = 10,
                 L: int = 5,
                 num_bins=20,
                 random_seed=1010,
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
        self.random_state = np.random.RandomState(random_seed)
        # class-conditional pattern-conditional contingency table
        self.T = {}
        # must be integers starting at 0 and increasing by 1
        self.classes_ = set(self.y)
        self.keep_copy_of_X = keep_copy_of_X
        self.num_bins = num_bins
        self.X_discretized = None  # type: np.array
        self.pattern_sets = {}
        self.feature_bins = {}
        self.fitted = False
        # new attributes as per convo with haralick
        self.nd_contingency_table = {}

    def predict(self, new_X: np.array = None):
        """
        Exposed API to make predictions
        :param new_X: New set of X values to make predictions with (optional)
        :return:
        """
        sorted_classes = range(len(self.classes_))
        if new_X is None:
            if not self.keep_copy_of_X:
                raise ValueError("Must keep copy of X in order to make predictions")
            if not self.fitted:
                raise LookupError("Model must be trained in order to make predicitons")
            # T_hat = {m: {cls: 0. for cls in self.classes_} for m in self.T.keys()}
            T_hat = {cls: np.zeros((self.X_discretized.shape[0], self.M)) for cls in self.classes_}
            for i, x in enumerate(self.X_discretized):
                for m, pattern_set in self.pattern_sets.items():
                    x_subspace = tuple(x[pattern_set])
                    for cls in self.classes_:
                        try:
                            T_hat[cls][i, m] = self.T[m][cls][x_subspace]
                        except KeyError:
                            continue
            log_prob = np.vstack([np.log(T_hat[cls] + 0.00001).sum(axis=1) for cls in sorted_classes]).T
            return log_prob.argmax(axis=1)


        else:
            new_X_discretized = self._discretize_prediction_data(new_X)
            T_hat = {cls: np.zeros((new_X_discretized.shape[0], self.M)) for cls in self.classes_}
            for i, x in enumerate(new_X_discretized):
                for m, pattern_set in self.pattern_sets.items():
                    x_subspace = tuple(x[pattern_set])
                    for cls in self.classes_:
                        try:
                            T_hat[cls][i, m] = self.T[m][cls][x_subspace]
                        except KeyError:
                            continue
            log_prob = np.vstack([np.log(T_hat[cls] + 0.00001).sum(axis=1) for cls in sorted_classes]).T
            return log_prob.argmax(axis=1)


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
        self.X_discretized = X_discretized.astype(int)

    def fit_init(self):
        """
        Exposed API for training an n-tuple classifier
        :return:
        """
        self._discretize_training_data()

        # create pattern sets
        possible_dimensions = np.arange(self.X.shape[1])
        self.T = {m: {cls: {} for cls in self.classes_} for m in range(self.M)}
        # each m is a subspace
        for m in range(self.M):
            self.pattern_sets[m] = {}
            # L is the dimensionality of the subspace m
            self.pattern_sets[m] = [self.random_state.choice(possible_dimensions) for l in range(self.L)]

        for m in range(self.M):
            for cls in self.classes_:
                cls_idx = np.where(self.y == cls)[0]
                num_records_in_class = cls_idx.shape[0]
                pattern_hash_table = {}
                for row in cls_idx:
                    x = self.X_discretized[row, :]
                    x_subspace = tuple(x[self.pattern_sets[m]])
                    try:
                        pattern_hash_table[x_subspace] += 1
                    except KeyError:
                        pattern_hash_table[x_subspace] = 1
                # get the class-conditional probability of a pattern
                self.T[m][cls] = {k: v / num_records_in_class for k, v in pattern_hash_table.items()}
        self.fitted = True


if __name__ == "__main__":
    import numpy as np
    from sklearn import datasets, metrics, cross_validation
    from midi_ml.models.n_tuple import *

    X, y = datasets.make_classification(n_samples=1000,
                                        n_features=10,
                                        n_informative=8,
                                        n_redundant=2,
                                        random_state=1010)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y)

    nt = NTupleClassifier(X_train, y_train, M=50, L=5, num_bins=10)
    nt.fit_init()
    nt.predict()
    preds = nt.predict(new_X=X_test)
    print(metrics.confusion_matrix(y_train, nt.predict()))
    print(metrics.confusion_matrix(y_test, preds))
import numpy as np
from functools import partial

# TODO move this to the unit tests
from sklearn import datasets


# TODO: unit tests
# TODO: change np.linalg.inv to np.linalg.solve

class PenalizedLogisticRegression(object):
    """
    Logistic Regression with an L2 penalty
    """

    def __init__(self,
                 X: np.array,
                 y: np.array,
                 l2_penalty: float = 0.,
                 num_iter: int = 10):
        """
        :param X: (N * M)-dimensional array containing the input data in matrix form
        :param y: (N * 1)-dimensional array containing the binary target variable, encoded as 0 and 1
        :param l2_penalty: The regularization parameter that enforces sparseness in the coefficients of our model
        :param num_iter: The number of iterations to use in running the Newton-Raphson algorithm
        """
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        self.X = X
        self.y = y
        self.lmbda_ = l2_penalty
        self.beta_ = np.zeros((1, X.shape[1]))  # np.random.uniform(-1, 1, size=(1, X.shape[1]))
        self.num_iter_ = num_iter
        self.log_likelihood_ = []

    def _log_likelihood(self) -> float:
        """
        Return the log-likelihood of the model given the data (i.e. P(D|M))
        :return:
        """
        theta_dot_x = self.beta_.dot(self.X.T)
        log_prob_data_given_theta = self.y * theta_dot_x - np.log(1 + np.exp(theta_dot_x))
        return np.sum(log_prob_data_given_theta)

    def predict_probabilities(self, new_X: np.array = None) -> np.array:
        """
        Return the vector of probability predictions
        :param new_X:
        :return:
        """
        if new_X is None:
            exp_log_odds = np.exp(np.dot(self.beta_, self.X.T)).T
        else:
            new_X = np.hstack([new_X, np.ones((new_X.shape[0], 1))])
            exp_log_odds = np.exp(np.dot(self.beta_, new_X.T)).T
        return (1. / (1. + exp_log_odds)).ravel()

    def _score_function(self, probabilities) -> np.array:
        """
        The gradient of the log-likelihood function
        :param probabilities: (N * 1)-dimensional array of inferred probabilities
        :return: (1 * M+1)-dimensional array
        """
        return np.dot(self.X.T, (self.y - probabilities)) + self.lmbda_ * self.beta_

    def _hessian(self, probabilities: np.array) -> np.array:
        """
        The second derivative of the log-likelihood function
        :param probabilities: (N * 1)-dimensional array of inferred probabilities
        :return: (M+1 * M+1)-dimensional array
        """
        W = np.eye(self.X.shape[0])
        for i in range(W.shape[0]):
            W[i, i] = probabilities[i] * (1 - probabilities[i])
        return self.X.T.dot(W).dot(self.X) + self.lmbda_ * np.eye(self.X.shape[1])

    def _newton_step(self) -> np.array:
        """
        Take one Newton-Raphson step. That is, move to where the tangent line of the
        first derivative of the log-likelihood function (score function) is equal to 0.
        :return: (M+1 * 1)-dimensional array containing the new parameters of the model (self.beta_)
        """
        probs = self.predict_probabilities()
        score = self._score_function(probs)
        hess = self._hessian(probs)
        step = np.linalg.inv(hess).dot(score.T).ravel()
        return (self.beta_ - step).reshape(self.beta_.shape)

    def _newton_raphson(self):
        """
        Run self.num_iter_ iterations of the Newton-Raphson algorithm to fit a logistic regression model
        :return:
        """
        for c in range(self.num_iter_):
            print("Running iteration {num} of Newton-Raphson algorithm".format(num=str(c)))
            self.log_likelihood_.append(self._log_likelihood())
            self.beta_ = self._newton_step()
        print("Model-fitting complete")

    def fit(self):
        """
        Fit a penalized logistic regression model
        :return:
        """
        self._newton_raphson()


import pdb


class LinearDiscriminantAnalysis(object):
    """

    """

    def __init__(self,
                 X: np.array,
                 y: np.array,
                 keep_copy_of_X=False):
        """
        :param X: (N * M)-dimensional array containing the input data in matrix form
        :param y: (N * 1)-dimensional array containing the binary target variable, encoded as 0 and 1
        """
        self.X = X
        self.y = y
        self.classes_ = set(self.y)
        self.keep_copy_of_X = keep_copy_of_X

        self.X_given_class_ = {}
        self.mean_given_class_ = {}
        self.class_priors_ = {}
        self.class_covariances_ = {}
        self.within_class_covariance_ = None  # type: np.array
        self.transformation_matrix_ = None  # type: np.array

    def _get_class_conditionals(self):
        """
        Separate the data by class, get the conditional means and class priors
        :return:
        """
        for c in self.classes_:
            self.X_given_class_[c] = self.X[np.where(self.y == c)]
            self.mean_given_class_[c] = self.X_given_class_[c].mean(axis=0)
            self.class_priors_[c] = self.X_given_class_[c].shape[0] / float(self.X.shape[0])
        if not self.keep_copy_of_X:
            self.X = None

    def _get_class_covariances(self):
        """
        Get the class conditional covariance matrices and the within-class covariance matrix
        :return:
        """
        self.within_class_covariance_ = np.zeros((self.X_given_class_[0].shape[1], self.X_given_class_[0].shape[1]))
        for c in self.classes_:
            X_minus_mu = self.X_given_class_[c] - self.mean_given_class_[c]
            self.class_covariances_[c] = np.dot(X_minus_mu.T, X_minus_mu) / (X_minus_mu.shape[0] - 1)
            self.within_class_covariance_ += self.class_priors_[c] * self.class_covariances_[c]

    def fit(self):
        self._get_class_conditionals()
        self._get_class_covariances()
        self.transformation_matrix_ = np.dot(np.linalg.inv(self.within_class_covariance_),
                                             (self.mean_given_class_[0] - self.mean_given_class_[1]))


# TODO move these into the NaiveBayesClassifier class
# TODO log_gaussian_pdf so we select the class that maximizes log-likelihood
def gaussian_pdf(x: float, mu: float, sigma: float) -> np.array:
    return np.array(1. / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)))


def log_gaussian_pdf(x: float, mu: float, sigma: float) -> np.array:
    return np.array(np.log(1. / np.sqrt(2 * np.pi * sigma ** 2)) - (x - mu) ** 2 / (2 * sigma ** 2))


def log_multinomial_pmf(counts: np.array, smoothing: int = 1):
    """

    :param counts: numpy array of k integers specifying the count of successes
                    across k categories
    :param smoothing: smoothing parameter from the Dirichlet prior. Setting this
                        to 1 is equivalent to Laplace Smoothing.
    :return:
    """
    n = counts.sum()
    return (counts + smoothing) / (n + smoothing)


# TODO: get a better understanding of the smoothing
class NaiveBayesClassifier(object):
    """
    Classifiers of the Naive Bayes family. All input features are assumed to be drawn from
     the same family of distributions.
     http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
    """

    # TODO: finish docstrings
    def __init__(self,
                 X: np.array,
                 y: np.array,
                 feature_family: str = "multinomial",
                 smoothing: int = 1,
                 keep_copy_of_X: bool = True):
        """
        :param X: (N * M)-dimensional array containing the input data in matrix form
        :param y: (N * 1)-dimensional array containing the binary target variable, encoded as 0 and 1
        """
        feature_families = ["gaussian", "multinomial", "bernoulli"]
        if feature_family not in feature_families:
            raise ValueError("Please select a distribution in %s" % str(feature_families))
        self.X = X
        self.y = y
        self.classes_ = set(self.y)
        self.feature_family_ = feature_family
        self.keep_copy_of_X = keep_copy_of_X
        self.smoothing_ = smoothing
        self.num_records_ = None  # type: int
        self.X_given_class_ = {}
        self.log_pdf_given_class_ = {}
        self.thetas_ = {}

    def _get_class_conditional_data(self):
        """
        Separate the data by class
        :return:
        """
        for c in self.classes_:
            self.X_given_class_[c] = self.X[np.where(self.y == c)]
        self.num_records_ = self.X.shape[0]
        if not self.keep_copy_of_X:
            self.X = None

    def _make_predictions(self, X: np.array = None):
        """

        :param X:
        :return:
        """
        predictions = np.zeros((self.X.shape[0], len(self.classes_)))
        for c in self.classes_:
            if self.feature_family_ in ("multinomial", "bernoulli"):
                class_conditional_log_probabilities = np.dot(X, self.thetas_[c])
            elif self.feature_family_ == "gaussian":
                class_conditional_log_probabilities = self.log_pdf_given_class_[c](self.X).sum(axis=1)
            else:
                raise ValueError("Must select proper feature family to make predictions")
            predictions[:, c] = class_conditional_log_probabilities
        return predictions.argmax(axis=1)

    def predict(self, new_X: np.array = None) -> np.array:
        """

        :param new_X:
        :return:
        """
        if new_X is None:
            if not self.keep_copy_of_X:
                raise ValueError("Must keep copy of X in order to make predictions")
            return self._make_predictions(self.X)
        else:
            return self._make_predictions(new_X)

    def _train_gaussian_model(self):
        """

        :return:
        """
        for c in self.classes_:
            means = self.X_given_class_[c].mean(axis=0)
            variances = self.X_given_class_[c].var(axis=0)
            self.log_pdf_given_class_[c] = partial(log_gaussian_pdf,
                                                   mu=means,
                                                   sigma=np.sqrt(variances))

    def _train_multinomial_model(self):
        """

        :return:
        """
        # We use the log of the probability that a document is drawn from this parametric
        # form of the distribution to ease the computation (by avoiding multiplying very small numbers)
        for c in self.classes_:
            # get values of n
            feature_sums = self.X_given_class_[c].sum(axis=0)
            alpha_i = float(self.smoothing_) / self.X_given_class_[c].shape[1]
            alpha = self.smoothing_
            self.thetas_[c] = np.log(feature_sums + alpha_i) - np.log(feature_sums.sum() + alpha)

    def _train_bernoulli_model(self):
        """

        :return:
        """
        # We use the log of the probability that a document is drawn from this parametric
        # form of the distribution to ease the computation (by avoiding multiplying very small numbers)
        for c in self.classes_:
            class_size = self.X_given_class_[c].shape[0]
            feature_counts = self.X_given_class_[c].sum(axis=0)
            self.thetas_[c] = np.log(feature_counts + 1) - np.log(class_size + 2)

    def _get_parametric_probability_estimates(self, feature_family: str):
        """

        :param feature_family:
        :return:
        """
        if feature_family == "bernoulli":
            self._train_bernoulli_model()
        elif feature_family == "multinomial":
            self._train_multinomial_model()
        elif feature_family == "gaussian":
            self._train_gaussian_model()
        else:
            raise ValueError("Please select a valid family of probability distributions")

    def fit(self):
        self._get_class_conditional_data()
        self._get_parametric_probability_estimates(feature_family=self.feature_family_)


def main():
    X, y = datasets.make_classification(n_samples=1000,
                                        n_features=3,
                                        n_informative=3,
                                        n_redundant=0)
    plr = PenalizedLogisticRegression(X=X, y=y, l2_penalty=5)
    lda = LinearDiscriminantAnalysis(X=X, y=y)
    nb = NaiveBayesClassifier(X=X, y=y, feature_family="gaussian")
    nb.fit()
    preds = nb.predict()
    from sklearn.metrics import confusion_matrix
    print(confusion_matrix(y, preds))
    X = np.random.multinomial(n=20,
                              pvals=np.random.dirichlet([1] * 10, 1).ravel(),
                              size=1000)
    nb = NaiveBayesClassifier(X=X, y=y, feature_family="multinomial")
    nb.fit()
    preds = nb.predict()
    print(confusion_matrix(y, preds))
    X = (X > 0).astype(int)
    nb = NaiveBayesClassifier(X=X, y=y, feature_family="bernoulli")
    nb.fit()
    preds = nb.predict()
    print(confusion_matrix(y, preds))
    pdb.set_trace()

    # plr.fit()
    # lda.fit()
    nb.fit()

    pdb.set_trace()


if __name__ == "__main__":
    main()

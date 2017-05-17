import logging
import numpy as np
from functools import partial

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")


# TODO: l1 penalty?

class PenalizedLogisticRegression(object):
    """
    Logistic Regression with an L2 penalty
    """

    def __init__(self,
                 X: np.array,
                 y: np.array,
                 l2_penalty: float = 0.,
                 num_iter: int = 10,
                 tol: float = 10e-5,
                 save_learning_info: bool = False):
        """
        :param X: (N * M)-dimensional array containing the input data in matrix form
        :param y: (N * 1)-dimensional array containing the binary target variable, encoded as 0 and 1
        :param l2_penalty: The regularization parameter that enforces sparseness in the coefficients of our model
        :param num_iter: The number of iterations to use in running the Newton-Raphson algorithm
        :param tol: The convergence tolerance
        :param save_learning_info: Whether or not to save the coefs and steps at each iteration
        """
        X = np.hstack([X, np.ones((X.shape[0], 1))])
        self.X = X
        self.y = y
        self.lmbda_ = l2_penalty
        self.beta_ = np.zeros((1, X.shape[1]))
        self.num_iter_ = num_iter
        self.tol_ = tol
        self.save_learning_info = save_learning_info
        self.converged_ = False
        self._log_likelihood_each_iter = []
        self._betas_each_iter = []
        self._step_each_iter = []

    # TODO: update computation to include regularization term on coefs
    def _log_likelihood(self) -> float:
        """
        Return the log-likelihood of the model given the data (i.e. P(D|M))
        :return:
        """
        beta_dot_x = self.beta_.dot(self.X.T)
        log_prob_data_given_theta = self.y * beta_dot_x - np.log(1 + np.exp(beta_dot_x)) - \
                                    self.lmbda_ * np.sum(self.beta_ ** 2)
        return np.sum(log_prob_data_given_theta)

    def predict_probabilities(self, new_X: np.array = None) -> np.array:
        """
        Return the vector of probability predictions
        :param new_X: unseen observations
        :return: array of predicted probabilities
        """
        if new_X is None:
            exp_neg_log_odds = np.exp(-np.dot(self.beta_, self.X.T)).T
        else:
            new_X = np.hstack([new_X, np.ones((new_X.shape[0], 1))])
            exp_neg_log_odds = np.exp(-np.dot(self.beta_, new_X.T)).T
        return (1. / (1. + exp_neg_log_odds)).ravel()

    def _score_function(self, probabilities) -> np.array:
        """
        The gradient of the log-likelihood function
        :param probabilities: (N * 1)-dimensional array of inferred probabilities
        :return: (1 * M+1)-dimensional array
        """
        return np.dot(self.X.T, (-self.y + probabilities)) + self.lmbda_ * self.beta_

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
        step = np.linalg.solve(hess, score.T).ravel()
        if self.save_learning_info:
            self._step_each_iter.append(step)
        new_betas = (self.beta_ - step).reshape(self.beta_.shape)
        return new_betas

    def _newton_raphson(self):
        """
        Run self.num_iter_ iterations of the Newton-Raphson algorithm to fit a logistic regression model
        :return:
        """
        self._log_likelihood_each_iter.append(self._log_likelihood())
        if self.save_learning_info:
            self._betas_each_iter.append(self.beta_)
        for c in range(self.num_iter_):
            self.beta_ = self._newton_step()
            self._log_likelihood_each_iter.append(self._log_likelihood())
            if self.save_learning_info:
                self._betas_each_iter.append(self.beta_)
            if (self._log_likelihood_each_iter[-1] - self._log_likelihood_each_iter[-2]) <= self.tol_:
                self.converged_ = True
                break
        print("Model-fitting complete")

    def fit(self):
        """
        Exposed API to fit a penalized logistic regression model
        :return:
        """
        self._newton_raphson()


class LinearDiscriminantAnalysis(object):
    """
    Train a two-class linear discriminant analysis classifier
    """

    def __init__(self,
                 X: np.array,
                 y: np.array,
                 regularization: float = None,
                 keep_copy_of_X=True):
        """
        :param X: (N * M)-dimensional array containing the input data in matrix form
        :param y: (N * 1)-dimensional array containing the binary target variable, encoded as 0 and 1
        :param keep_copy_of_X: whether or not to persist a copy of the data in the model object
        """
        self.X = X
        self.y = y
        self.classes_ = set(self.y)
        self.keep_copy_of_X = keep_copy_of_X

        self.X_given_class_ = {}
        self.mean_given_class_ = {}
        self.class_priors_ = {}
        self.class_covariances_ = {}
        self.decision_threshold_ = None  # type: float
        self.within_class_covariance_ = None  # type: np.array
        self.between_class_covariance_ = None  # type: np.array
        self.transformation_matrix_ = None  # type: np.array
        self.projected_means_ = {}
        self.regularization_ = regularization

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

    def predict(self, new_X: np.array = None):
        """
        Exposed API to make predictions. Points are classified based on distance to projected mean
        in the projected space
        :param new_X: New set of X values to make predictions with (optional)
        :return:
        """
        if new_X is None:
            if not self.keep_copy_of_X:
                raise ValueError("Must keep copy of X in order to make predictions")
            projected_data = self.X.dot(self.transformation_matrix_)
        else:
            projected_data = new_X.dot(self.transformation_matrix_)
        dist_to_means = [((projected_data - self.projected_means_[cls]) ** 2).sum(axis=1) for cls in self.classes_]
        return np.stack(dist_to_means).T.argmin(axis=1)

    def fit(self):
        """
        Exposed API for training a LDA model
        :return:
        """
        self._get_class_conditionals()
        self._get_class_covariances()
        if self.regularization_ is not None:
            reg_covariance = self.regularization_ * self.within_class_covariance_
            reg_covariance += (1 - self.regularization_) * np.eye(self.within_class_covariance_.shape[1])
            self.within_class_covariance_ = reg_covariance
        num_features = self.X.shape[1]
        global_means = self.X.mean(axis=0).reshape((num_features, 1))
        self.between_class_covariance_ = np.zeros((num_features, num_features))
        for cls in self.classes_:
            class_means = self.mean_given_class_[cls].reshape((num_features, 1))
            class_counts = self.X_given_class_[cls].shape[0]
            class_means_dot_global = (class_means - global_means)
            self.between_class_covariance_ += class_counts * class_means_dot_global.dot(class_means_dot_global.T)
        self.transformation_matrix_ = np.linalg.solve(self.within_class_covariance_, self.between_class_covariance_)
        proj_means = {cls: self.mean_given_class_[cls].dot(self.transformation_matrix_) for cls in self.classes_}
        self.projected_means_ = proj_means


class NaiveBayesClassifier(object):
    """
    Classifiers of the Naive Bayes family. All input features are assumed to be drawn from
     a distribution of the same parametric form
     http://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf
    """

    def __init__(self,
                 X: np.array,
                 y: np.array,
                 parametric_form: str = "multinomial",
                 keep_copy_of_X: bool = True,
                 smoothing: int = 1):
        """
        :param X: (N * M)-dimensional array containing the input data in matrix form
        :param y: (N * 1)-dimensional array containing the binary target variable, encoded as 0 and 1
        :param parametric_form: the parametric form that the features are assumed to take (used to define PDF)
        :param keep_copy_of_X: Whether to keep a copy of X in memory (to be used for making predictions on the training set)
        :param smoothing: Smoothing parameter for dirichlet prior in multinomial model (sets to None if "bernoulli" or "gaussian" or chosen)
        """
        parametric_forms = ["bernoulli", "multinomial", "gaussian"]
        if parametric_form not in parametric_forms:
            raise ValueError("Please select a distribution in %s" % str(parametric_forms))
        self.X = X
        self.y = y
        self.classes_ = set(self.y)
        self.parametric_form_ = parametric_form
        self.keep_copy_of_X = keep_copy_of_X
        if parametric_form is "multinomial":
            self.smoothing_ = smoothing
        else:
            self.smoothing_ = None
        self.num_records_ = None  # type: int
        self.X_given_class_ = {}
        self.thetas_ = {}
        self.priors_ = {}
        self.log_pdf_given_class_ = {}

    @staticmethod
    def log_gaussian_pdf(x: float, mu: float, sigma: float) -> np.array:
        """
        Log of the Gaussian probability density function
        :param x: value (or np.array of values) at which we compute the relative log-likelihood of drawing that point
        :param mu: mean of the Gaussian
        :param sigma: standard deviation of the Gaussian
        :return: Array with the log-probability of each x
        """
        return np.array(np.log(1. / np.sqrt(2 * np.pi * sigma ** 2)) - (x - mu) ** 2 / (2 * sigma ** 2))

    def _get_class_conditional_data(self):
        """
        Separate the data by class
        :return:
        """
        self.num_records_ = self.X.shape[0]
        for c in self.classes_:
            self.X_given_class_[c] = self.X[np.where(self.y == c)]
            self.priors_[c] = float(self.X_given_class_[c].shape[0]) / self.num_records_
        if not self.keep_copy_of_X:
            self.X = None

    def _make_predictions(self, X: np.array = None):
        """
        Predict the class of the input values X
        :param X: matrix to make predictions with
        :return:
        """
        predictions = np.zeros((X.shape[0], len(self.classes_)))
        for c in self.classes_:
            if self.parametric_form_ in ("multinomial", "bernoulli"):
                class_conditional_log_probabilities = np.dot(X, self.thetas_[c])
            elif self.parametric_form_ == "gaussian":
                class_conditional_log_probabilities = np.nan_to_num(self.log_pdf_given_class_[c](X)).sum(axis=1)
            else:
                raise ValueError("Must select proper feature family to make predictions")
            # we add the log of the prior probability of the class as an "intercept"
            predictions[:, c] = class_conditional_log_probabilities + np.log(self.priors_[c])
        return predictions.argmax(axis=1)

    def predict(self, new_X: np.array = None) -> np.array:
        """
        Exposed API to make predictions
        :param new_X: New set of X values to make predictions with (optional)
        :return:
        """
        if new_X is None:
            if not self.keep_copy_of_X:
                raise ValueError("Must keep copy of X in order to make predictions")
            return self._make_predictions(self.X)
        else:
            return self._make_predictions(new_X)

    def _train_bernoulli_model(self):
        """
        Train a Bernoulli Naive Bayes
        :return:
        """
        # We use the log of the probability that a document is drawn from this parametric
        # form of the distribution to ease the computation (by avoiding multiplying very small numbers)
        for c in self.classes_:
            class_size = self.X_given_class_[c].shape[0]
            feature_counts = self.X_given_class_[c].sum(axis=0)
            self.thetas_[c] = np.log(feature_counts + 1) - np.log(class_size + 2)

    def _train_multinomial_model(self):
        """
        Train a Multinomial Naive Bayes
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

    # TODO: in high dimensions we'll be storing 10s of thousands of functions which could be inefficient
    def _train_gaussian_model(self):
        """
        Train a Gaussain Naive Bayes
        :return:
        """
        for c in self.classes_:
            means = self.X_given_class_[c].mean(axis=0)
            variances = self.X_given_class_[c].var(axis=0)
            self.log_pdf_given_class_[c] = partial(self.log_gaussian_pdf,
                                                   mu=means,
                                                   sigma=np.sqrt(variances))

    def _get_parametric_probability_estimates(self, parametric_form: str):
        """
        Estimate the parameters of the parametric probability distributions
        :param parametric_form: the form we assume for estimating PDFs/PMFs
        :return:
        """
        if parametric_form == "bernoulli":
            self._train_bernoulli_model()
        elif parametric_form == "multinomial":
            self._train_multinomial_model()
        elif parametric_form == "gaussian":
            self._train_gaussian_model()
        else:
            raise ValueError("Please select a valid family of probability distributions")

    def fit(self):
        """
        Exposed API for training model
        :return:
        """
        self._get_class_conditional_data()
        self._get_parametric_probability_estimates(parametric_form=self.parametric_form_)


def main():
    from sklearn import datasets
    X, y = datasets.make_classification(n_samples=1000,
                                        n_features=2,
                                        n_informative=2,
                                        n_redundant=0,
                                        random_state=1010)
    plr = PenalizedLogisticRegression(X=X, y=y, l2_penalty=15, num_iter=7)
    plr.fit()
    lda = LinearDiscriminantAnalysis(X=X, y=y)
    lda.fit()

    nb = NaiveBayesClassifier(X=X, y=y, parametric_form="gaussian")
    nb.fit()

    X = np.random.multinomial(n=20,
                              pvals=np.random.dirichlet([1] * 10, 1).ravel(),
                              size=1000)
    nb = NaiveBayesClassifier(X=X, y=y, parametric_form="multinomial")
    nb.fit()
    X = (X > 0).astype(int)
    nb = NaiveBayesClassifier(X=X, y=y, parametric_form="bernoulli")
    nb.fit()


if __name__ == "__main__":
    main()

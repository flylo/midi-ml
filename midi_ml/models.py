import numpy as np
from sklearn import datasets


class PenalizedLogisticRegression(object):
    """
    Logistic Regression with an L2 penalty
    """

    def __init__(self, X: np.array,
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
        self.beta_ = np.random.uniform(-1, 1, size=(1, X.shape[1]))
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
        for i in range(self.num_iter_):
            print("Running iteration {num} of Newton-Raphson algorithm".format(num=str(i)))
            self.log_likelihood_.append(self._log_likelihood())
            self.beta_ = self._newton_step()
        print("Model-fitting complete")

    def fit(self):
        """
        Fit a penalized logistic regression model
        :return:
        """
        self._newton_raphson()


def main():
    X, y = datasets.make_classification(n_samples=1000,
                                        n_features=3,
                                        n_informative=3,
                                        n_redundant=0)
    plr = PenalizedLogisticRegression(X=X, y=y, l2_penalty=5)
    plr.fit()
    print(plr.log_likelihood_)


if __name__ == "__main__":
    main()

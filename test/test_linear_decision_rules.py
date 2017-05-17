import unittest
import numpy as np
import operator
from sklearn import datasets
from sklearn.metrics import confusion_matrix, roc_auc_score
from midi_ml.models.linear_decision_rules import PenalizedLogisticRegression, LinearDiscriminantAnalysis, \
    NaiveBayesClassifier

# TODO: remove
import pdb


def subsequent_entry_relationships(x: list, op: operator = operator.ge):
    """
    check the truth value of `op` for each subsequent entry in a list
    :param x: list of numbers
    :param op: operator to compare
    :return:
    """
    return all([op(a, b) for (a, b) in zip(x, x[1:])])


class PenalizedLogisticRegressionTestCase(unittest.TestCase):
    """
    Test cases for the PenalizedLogisticRegression model
    """

    X, y = datasets.make_classification(n_samples=1000,
                                        n_features=3,
                                        n_informative=3,
                                        n_redundant=0,
                                        random_state=1010)

    def test_model_penalization(self):
        """
        model should build and larger penalties should produce smaller coefficients
        :return:
        """
        sum_squared_coefs = []
        for penalty in np.logspace(-1, 2, 4):
            plr = PenalizedLogisticRegression(X=self.X, y=self.y, l2_penalty=penalty)
            plr.fit()
            sum_squared_coefs.append(np.sum(plr.beta_ ** 2))

        self.assertTrue(subsequent_entry_relationships(sum_squared_coefs))

    def test_newton_steps(self):
        """
        gradient information should be computed properly
        :return:
        """
        plr = PenalizedLogisticRegression(X=self.X, y=self.y, num_iter=20, save_learning_info=True)
        plr.fit()
        # all log-likelihoods should be strictly increasing
        self.assertTrue(subsequent_entry_relationships(plr._log_likelihood_each_iter, op=operator.le))
        # steps should be getting smaller
        l1_norm_steps = [sum(abs(s)) for s in plr._step_each_iter]
        self.assertTrue(subsequent_entry_relationships(l1_norm_steps, op=operator.ge))
        # model should stop running upon convergence
        self.assertTrue(plr.converged_ is True)
        self.assertTrue(len(plr._log_likelihood_each_iter) < 20)
        self.assertTrue((plr._log_likelihood_each_iter[-1] - plr._log_likelihood_each_iter[-2]) <= plr.tol_)

    def test_coefficients(self):
        """
        Beta coefficients should be what you'd expect
        :return:
        """
        np.random.seed(1010)
        # X_1 should be negatively correlated and X_2 should be positively correlated
        X1_0 = np.random.normal(loc=1, size=100)
        X2_0 = np.random.normal(loc=-1, size=100)
        X_0 = np.array([X1_0, X2_0]).T
        X1_1 = np.random.normal(loc=-1, size=100)
        X2_1 = np.random.normal(loc=1, size=100)
        X_1 = np.array([X1_1, X2_1]).T
        X = np.concatenate([X_0, X_1])
        y = np.concatenate([np.zeros((100,)), np.ones((100,))]).astype(int)

        # betas should match known correlation structures in the data
        plr = PenalizedLogisticRegression(X=X, y=y)
        plr.fit()
        beta = plr.beta_.ravel()
        self.assertTrue(beta[0] < 0)
        self.assertTrue(beta[1] > 0)

        # model should be predictive
        self.assertTrue(roc_auc_score(y, plr.predict_probabilities() > 0.5) > 0.9)


class LinearDiscriminantAnalysisTestCase(unittest.TestCase):
    """
    Test cases for the LinearDiscriminantAnalysis model
    """

    np.random.seed(1010)
    X1_0 = np.random.normal(loc=1, size=100)
    X2_0 = np.random.normal(loc=2, size=100)
    X_0 = np.array([X1_0, X2_0]).T

    X1_1 = np.random.normal(loc=-1, size=100)
    X2_1 = np.random.normal(loc=-2, size=100)
    X_1 = np.array([X1_1, X2_1]).T
    X = np.concatenate([X_0, X_1])
    y = np.concatenate([np.zeros((100,)), np.ones((100,))]).astype(int)
    lda = LinearDiscriminantAnalysis(X=X, y=y)
    lda.fit()

    def test_model_building(self):
        """
        test that the model builds without error and that it makes correct predictions
        :return:
        """
        preds = self.lda.predict()
        confusion = confusion_matrix(self.y, preds)
        self.assertTrue(confusion[0, 0] > 90)
        self.assertTrue(confusion[1, 1] > 90)

    def test_parameters(self):
        """
        test that we correctly compute the within-class means and priors
        :return:
        """
        self.assertTrue(all(np.round(self.lda.mean_given_class_[0]) == [1, 2]))
        self.assertTrue(all(np.round(self.lda.mean_given_class_[1]) == [-1, -2]))
        self.assertTrue(self.lda.class_priors_[0] == self.lda.class_priors_[0] == 0.5)

    def test_regularization(self):
        lda_reg = LinearDiscriminantAnalysis(X=self.X, y=self.y, regularization=0.7)
        lda_reg.fit()
        # Check that the sample covariance between features is smaller when we introduce regularization
        self.assertTrue(self.lda.within_class_covariance_[0, 1] - lda_reg.within_class_covariance_[0, 1] > 0)
        self.assertTrue(self.lda.within_class_covariance_[1, 0] - lda_reg.within_class_covariance_[1, 0] > 0)

    def test_multiple_casses(self):
        np.random.seed(1010)
        X1_0 = np.random.normal(loc=1, size=100)
        X2_0 = np.random.normal(loc=2, size=100)
        X_0 = np.array([X1_0, X2_0]).T
        X1_1 = np.random.normal(loc=-1, size=100)
        X2_1 = np.random.normal(loc=-2, size=100)
        X_1 = np.array([X1_1, X2_1]).T
        X1_2 = np.random.normal(loc=-10, size=100)
        X2_2 = np.random.normal(loc=-20, size=100)
        X_2 = np.array([X1_2, X2_2]).T
        X = np.concatenate([X_0, X_1, X_2])
        y = np.concatenate([np.zeros((100,)), np.ones((100,)), np.ones((100,)) + 1]).astype(int)
        lda = LinearDiscriminantAnalysis(X=X, y=y)
        lda.fit()
        confusion = confusion_matrix(y, lda.predict())
        self.assertTrue(confusion[0,0] > 90)
        self.assertTrue(confusion[1, 1] > 90)
        self.assertTrue(confusion[2, 2] > 90)


class NaiveBayesClassifierTestCase(unittest.TestCase):
    """
    Test cases for the NaiveBayesClassifier model
    """

    y = np.concatenate([np.zeros((100,)), np.ones((100,))]).astype(int)

    def test_gaussian_model(self):
        """
        test that the gaussian naive bayes classifier runs and produces sensible log probabilities
        :return:
        """
        np.random.seed(1010)
        X1_0 = np.random.normal(loc=1, size=100)
        X2_0 = np.random.normal(loc=2, size=100)
        X_0 = np.array([X1_0, X2_0]).T

        X1_1 = np.random.normal(loc=-1, size=100)
        X2_1 = np.random.normal(loc=-2, size=100)
        X_1 = np.array([X1_1, X2_1]).T
        X = np.concatenate([X_0, X_1])

        gaussian_nb = NaiveBayesClassifier(X, self.y, parametric_form="gaussian")
        gaussian_nb.fit()

        confusion = confusion_matrix(self.y, gaussian_nb.predict())

        self.assertTrue(gaussian_nb.priors_[0] == gaussian_nb.priors_[1] == 0.5)
        # sum of log prob of negative number given class 0 should be less than sum log prob of neg number given class 1
        self.assertTrue(gaussian_nb.log_pdf_given_class_[0](-1).sum() < gaussian_nb.log_pdf_given_class_[1](-1).sum())
        self.assertTrue(confusion[0, 0] > 90, confusion[1, 1] > 90)

        # predictions should work for new arrays
        gaussian_nb = NaiveBayesClassifier(X, self.y, parametric_form="gaussian", keep_copy_of_X=False)
        gaussian_nb.fit()
        random_subset = np.random.choice(np.arange(X.shape[0]), size=10)
        new_X = X[random_subset]
        self.assertTrue(gaussian_nb.predict(new_X=new_X) is not None)

    def test_multinomial_model(self):
        """
        test that the multinomial naive bayes classifier runs and produces sensible log probabilities
        :return:
        """
        np.random.seed(1010)
        X1_0 = np.random.choice(range(10), size=100)
        X2_0 = np.random.choice(range(2), size=100)
        X_0 = np.array([X1_0, X2_0]).T

        X1_1 = np.random.choice(range(2), size=100)
        X2_1 = np.random.choice(range(10), size=100)
        X_1 = np.array([X1_1, X2_1]).T
        X = np.concatenate([X_0, X_1])

        multinomial_nb = NaiveBayesClassifier(X, self.y, parametric_form="multinomial")
        multinomial_nb.fit()

        confusion = confusion_matrix(self.y, multinomial_nb.predict())

        self.assertTrue(multinomial_nb.priors_[0] == multinomial_nb.priors_[1] == 0.5)
        # log probability of X1 should be greater than log probability of X2 for class 0
        self.assertTrue(multinomial_nb.thetas_[0][0] > multinomial_nb.thetas_[0][1])
        # log probability of X2 should be greater than log probability of X1 for class 1
        self.assertTrue(multinomial_nb.thetas_[1][1] > multinomial_nb.thetas_[0][0])
        self.assertTrue(confusion[0, 0] > 90, confusion[1, 1] > 90)

        # predictions should work for new arrays
        multinomial_nb = NaiveBayesClassifier(X, self.y, parametric_form="multinomial", keep_copy_of_X=False)
        multinomial_nb.fit()
        random_subset = np.random.choice(np.arange(X.shape[0]), size=10)
        new_X = X[random_subset]
        self.assertTrue(multinomial_nb.predict(new_X=new_X) is not None)

    def test_bernoulli_model(self):
        """
        test that the bernoulli naive bayes classifier runs and produces sensible log probabilities
        :return:
        """
        np.random.seed(1010)
        X1_0 = np.random.choice(range(10), size=100)
        X2_0 = np.random.choice(range(2), size=100)
        X_0 = np.array([X1_0, X2_0]).T

        X1_1 = np.random.choice(range(2), size=100)
        X2_1 = np.random.choice(range(10), size=100)
        X_1 = np.array([X1_1, X2_1]).T
        X = np.concatenate([X_0, X_1])
        # force Xs to be bernoulli RVs (i.e. just 0s and 1s)
        X = (X > 0).astype(int)

        bernoulli_nb = NaiveBayesClassifier(X, self.y, parametric_form="bernoulli")
        bernoulli_nb.fit()

        confusion = confusion_matrix(self.y, bernoulli_nb.predict())

        self.assertTrue(bernoulli_nb.priors_[0] == bernoulli_nb.priors_[1] == 0.5)
        # log probability of X1 should be greater than log probability of X2 for class 0
        self.assertTrue(bernoulli_nb.thetas_[0][0] > bernoulli_nb.thetas_[0][1])
        # log probability of X2 should be greater than log probability of X1 for class 1
        self.assertTrue(bernoulli_nb.thetas_[1][1] > bernoulli_nb.thetas_[0][0])
        self.assertTrue(confusion[1, 1] > 90)

        # predictions should work for new arrays
        bernoulli_nb = NaiveBayesClassifier(X, self.y, parametric_form="bernoulli", keep_copy_of_X=False)
        bernoulli_nb.fit()
        random_subset = np.random.choice(np.arange(X.shape[0]), size=10)
        new_X = X[random_subset]
        self.assertTrue(bernoulli_nb.predict(new_X=new_X) is not None)

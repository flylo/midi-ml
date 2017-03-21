import unittest
# from unittest.mock import MagicMock
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from midi_ml.models.linear_decision_rules import PenalizedLogisticRegression


class PenalizedLogisticRegressionTestCase(unittest.TestCase):

    X, y = datasets.make_classification(n_samples=1000,
                                        n_features=3,
                                        n_informative=3,
                                        n_redundant=0,
                                        random_state=1010)

    def test_model_building(self):
        # model should build and larger penalties should produce smaller coefficients
        sum_squared_coefs = []
        for penalty in np.logspace(-1, 2, 4):
            plr = PenalizedLogisticRegression(X=self.X, y=self.y, l2_penalty=penalty)
            plr.fit()
            sum_squared_coefs.append(np.sum(plr.beta_**2))

        self.assertTrue([a >= b for (a,b) in zip(sum_squared_coefs, sum_squared_coefs[1:])])

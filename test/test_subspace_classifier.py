import unittest
import numpy as np
from sklearn import datasets
from midi_ml.models.subspace_classifier import SubspaceClassifier


class SubspaceClassifierTestCase(unittest.TestCase):
    np.random.seed(1010)
    X1_0 = np.random.normal(loc=10, size=100)
    X2_0 = np.random.normal(loc=4, size=100)
    X_0 = np.array([X1_0, X2_0]).T

    X1_1 = np.random.normal(loc=-1, size=100)
    X2_1 = np.random.normal(loc=-2, size=100)
    X_1 = np.array([X1_1, X2_1]).T
    X = np.concatenate([X_0, X_1])
    y = np.concatenate([np.zeros((100,)), np.ones((100,))]).astype(int)

    sc = SubspaceClassifier(X=X, y=y)
    sc.fit()

    def test_model_build(self):
        self.assertTrue(self.sc is not None)

    def test_redundant_data(self):
        X, y = datasets.make_classification(n_samples=1000,
                                            n_features=100,
                                            n_informative=2,
                                            n_redundant=98,
                                            random_state=1010)
        sc = SubspaceClassifier(X=X, y=y)
        sc.fit()
        self.assertTrue(sc is not None)

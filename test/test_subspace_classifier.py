import unittest
import numpy as np
from midi_ml.models.subspace_classifier import SubspaceClassifier

import pdb


class SubspaceClassifierTestCase(unittest.TestCase):
    # np.random.seed(1010)
    # X1_0 = np.random.normal(loc=1, size=100)
    # X2_0 = np.random.normal(loc=2, size=100)
    # X_0 = np.array([X1_0, X2_0]).T
    #
    # X1_1 = np.random.normal(loc=-1, size=100)
    # X2_1 = np.random.normal(loc=-2, size=100)
    # X_1 = np.array([X1_1, X2_1]).T
    # X = np.concatenate([X_0, X_1])
    # y = np.concatenate([np.zeros((100,)), np.ones((100,))]).astype(int)


    np.random.seed(1010)
    X1_0 = np.random.normal(loc=1, size=100)
    X2_0 = np.random.normal(loc=2, size=100)
    X_0 = np.array([X1_0, X2_0]).T
    X1_1 = np.random.normal(loc=-1, size=100)
    X2_1 = np.random.normal(loc=-2, size=100)
    X_1 = np.array([X1_1, X2_1]).T
    X = np.concatenate([X_0, X_1])
    sc = SubspaceClassifier(X=X, y=y)
    sc.fit()

    def test_model_build(self):
        self.assertTrue(self.sc is not None)
        pdb.set_trace()

import unittest
import numpy as np
from midi_ml.models.decomposition import PrincipalComponents


class PrincipalComponentsTestCase(unittest.TestCase):
    """
    Test cases for the PenalizedLogisticRegression model
    """
    np.random.seed(1010)
    correlated_features = np.random.multivariate_normal(np.array([0, 0]),
                                                        cov=np.array([[1, 0.9],
                                                                      [0.9, 1]]),
                                                        size=1000)
    uncorrelated_features = np.random.multivariate_normal(np.array([0, 0]),
                                                          cov=np.array([[1, 0],
                                                                        [0, 1]]),
                                                          size=1000)
    X = np.concatenate([correlated_features, uncorrelated_features], axis=1)

    def test_principle_components_fit(self):
        """
        Test that the eigenvalue decompositions behaves as expected
        :return:
        """
        pc = PrincipalComponents(self.X)
        pc.fit()
        # We have 2 highly correlated features and 2 features that are perfectly uncorrelated.
        # Therefore, one of our PCs should have a corresponding eigenvalue of ~2 and the next two
        # PCs should have eigenvalues of ~1.
        self.assertTrue(np.allclose(pc.eigenvalues_, np.array([2, 1, 1, 0]), atol=0.2))
        self.assertTrue(pc.transform() is not None)

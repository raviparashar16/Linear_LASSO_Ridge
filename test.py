"""
A short testing program using the unittest (AKA PyUnit) module in Python. It tests the LinReg class - specifically
the fit and predict methods. All tests passing indicates that the methods are robust to multiple erroneous inputs.
"""

__author__ = 'Ravi Parashar'
__version__ = '1.0'

# necessary imports for tests
import numpy as np
import unittest
from LinReg import LinearRegressionModel


# testing the fit method of LinearRegressionModel
class FitTestCase(unittest.TestCase):
    # common code for the tests
    def setUp(self):
        # linear regression model instantiated
        self.model = LinearRegressionModel()

    # test to see if program exits when y has more than one column and when it has more rows than X
    def test_y_dimensions(self):
        y = np.ones((100, 2))
        X = np.ones((100, 5))
        with self.assertRaises(SystemExit):
            self.model.fit(X, y)
        X = np.ones((99, 5))
        with self.assertRaises(SystemExit):
            self.model.fit(X, y)

    # test to see if program exits when X matrix has inconsistent row lengths and to see if it does not exit with
    # consistent row lengths
    def test_constant_x_rows(self):
        y = [1, 2, 3, 4]
        X = [[1, 2, 3, 4], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4]]
        with self.assertRaises(SystemExit):
            self.model.fit(X, y)
        X = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        try:
            self.model.fit(X, y)
        except ValueError:
            self.fail("Unexpected error.")

    # test to see if program exits when either X matrix or y vector have a non-numeric element
    def test_x_y_nan(self):
        y = [1, 2, 3, 4]
        X = [[1, 2, 'c', 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        with self.assertRaises(SystemExit):
            self.model.fit(X, y)
        y = [1, 2, 'c', 4]
        X = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        with self.assertRaises(SystemExit):
            self.model.fit(X, y)

    # test to see if program exits when an improper method argument is used and does not exit when a proper method
    # argument is used
    def test_wrong_method(self):
        y = [1, 2, 3, 4]
        X = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        with self.assertRaises(SystemExit):
            self.model.fit(X, y, "logistic")
        try:
            self.model.fit(X, y, "LINEAR")
        except ValueError:
            self.fail("Unexpected error.")


# testing the predict method of LinearRegressionModel
class PredictTestCase(unittest.TestCase):
    def setUp(self):
        self.model = LinearRegressionModel()
        X_train = np.ones((100, 5))
        y_train = np.ones((100, ))
        self.model.fit(X_train, y_train)

    # tests to see if program exits when X matrix has inconsistent row lengths and to see if it does not exit with
    # consistent row lengths
    def test_constant_x_rows(self):
        X = [[1, 2, 3, 4, 5], [1, 2, 3], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        with self.assertRaises(SystemExit):
            self.model.predict(X)

    # tests to see if program exits when the model has not yet been fitted on data
    def test_model_not_fitted(self):
        X = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        model1 = LinearRegressionModel()
        with self.assertRaises(SystemExit):
            model1.predict(X)

    # tests to see if program exits when the row length of the test X matrix does not match with the row length of the
    # train X matrix
    def test_coefficient_mismatch(self):
        X = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        with self.assertRaises(SystemExit):
            self.model.predict(X)

    # test to see if program exits when X matrix has a non-numeric element
    def test_x_nan(self):
        X = [[1, 2, 3, 4, 5], [1, 2, 'c', 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        with self.assertRaises(SystemExit):
            self.model.predict(X)

    # test to see if program does not exit with proper input
    def test_normal(self):
        X = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]]
        try:
            self.model.predict(X)
        except ValueError:
            self.fail("Unexpected error.")


# tests executed
if __name__ == '__main__':
    unittest.main()

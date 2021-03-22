"""
This program contains the linear regression model class. Each model instance can be fit to input data using normal
multivariate linear regression or a LASSO/ridge penalty can be added. Predictions can be calculated by using the predict
function with feature values as input. The main method contains a demo using the Boston housing dataset.
"""

__author__ = 'Ravi Parashar'
__version__ = '1.0'

# necessary imports for functions, class, and main (demo) function
import numpy as np
import numbers
from sklearn.datasets import load_boston
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split


# soft threshold gradient formula for LASSO regression
def soft_thresh(x, b, j, lam, y):
    # jth component of b vector
    bj = b[j]
    # if bj < 0, derivative of lasso penalty portion of cost function is -lambda
    # if bj > 0, it is +lambda
    # if bj = 0, it is 0
    if bj < 0:
        return np.matmul(np.transpose(x[:, j]), np.subtract(np.matmul(x, b), y)) - lam
    if bj == 0:
        return np.matmul(np.transpose(x[:, j]), np.subtract(np.matmul(x, b), y))
    if bj > 0:
        return np.matmul(np.transpose(x[:, j]), np.subtract(np.matmul(x, b), y)) + lam


# class for a model
class LinearRegressionModel:
    # constructor with empty coefficient list
    def __init__(self):
        self.coeff = []

    """
    function to fit model to training data
    default method for fitting is linear (as opposed to LASSO/Ridge), default lambda is 0 (which is only used
    in the case of LASSO/Ridge fitting method)
    default parameters for optimization are epochs = 50000, update threshold = 0.0001, and learning rate = 0.03
    """
    def fit(self, matrix_x, vector_y, method="linear", lam=0, epochs=50000, update_thresh=0.0001, learn_rate=0.03):
        matrix_x = np.asarray(matrix_x)
        vector_y = np.asarray(vector_y)
        # dimensions of feature matrix determined; if it does not match the expected dimensions, an error message is
        # printed and the program is immediately quit
        if len(matrix_x.shape) != 2:
            print("Error. The input X matrix should consist of 2 dimensions and constant row and column lengths.")
            quit()
        row_x, col_x = matrix_x.shape
        # dimension of output vector determined; if it does not match the expected dimensions, an error message is
        # printed and the program is immediately quit
        if len(vector_y.shape) != 1:
            print("Error. The input y vector should consist of 1 column.")
            quit()
        row_y = vector_y.shape[0]
        # error flag initially set to false
        err_flag = False
        # if the first element of the X matrix is not of type np.ndarray it likely means that the input matrix did not
        # have constant row lengths; an error message is printed and error flag raised to true
        if not isinstance(matrix_x[0], np.ndarray):
            print("Error. The input X matrix should have constant row lengths.")
            err_flag = True
        # if the row size of x and y are not equal an error message is printed and error flag is raised to true
        if row_x != row_y:
            print("Error. The X matrix and y vector should have the same number of rows.")
            err_flag = True
        # if any of the values in the X matrix or the y vector are not numeric, an error message is printed and error
        # flag is raised to true
        isnan = False
        temp_x = matrix_x.flatten()
        for i in temp_x:
            if not isinstance(i, numbers.Number):
                isnan = True
                break
        if isnan:
            print("Error. At least one of the values in the X matrix is non-numeric.")
            err_flag = True
        isnan = False
        temp_y = vector_y.flatten()
        for i in temp_y:
            if not isinstance(i, numbers.Number):
                isnan = True
                break
        if isnan:
            print("Error. At least one of the values in the y vector is non-numeric.")
            err_flag = True
        # if the error flag was raised to true at any point, the program is quit
        if err_flag:
            quit()
        # a column of ones is prepended to the X matrix to account for the 0th component of the coefficient vector
        # then the column size of x is increased by 1
        add_col_x = np.ones(row_x)
        matrix_x = np.column_stack((matrix_x, add_col_x))
        col_x += 1
        # b vector is created with length of number of columns in X matrix
        b = np.ones(col_x)
        # number of epochs before termination of optimization
        num_epochs = epochs
        # what the maximum update in the b vector must be under in order to terminate optimization
        update_threshold = update_thresh
        # learning rate
        eta = learn_rate
        # if it is linear regression with no regularization
        if method.lower() == "linear":
            prev_b = b
            # optimize for num_epoch epochs
            for curr_epoch in range(num_epochs):
                # if the current epoch is not the first epoch and the maximum update in the b vector is less than
                # update_threshold, terminate optimization
                if curr_epoch > 0:
                    diff_b = np.abs(np.subtract(b, prev_b))
                    if np.amax(diff_b) < update_threshold:
                        self.coeff = b
                        return
                # previous b vector set to this b vector
                prev_b = b
                # b vector updated
                b = np.subtract(b, ((eta/col_x) * np.matmul(matrix_x.transpose(), np.subtract(np.matmul(matrix_x, b),
                                                                                              vector_y))))
            # after num_epochs reached, model's coeff instance variable set to b
            self.coeff = b
            return
        # if it is ridge regression
        if method.lower() == "ridge":
            prev_b = b
            for curr_epoch in range(num_epochs):
                if curr_epoch > 0:
                    diff_b = np.abs(np.subtract(b, prev_b))
                    if np.amax(diff_b) < update_threshold:
                        self.coeff = b
                        return
                prev_b = b
                b = np.subtract(b, ((eta/col_x) * (np.matmul(matrix_x.transpose(), np.subtract(np.matmul(matrix_x, b),
                                                                                               vector_y)) + (lam * b))))
            self.coeff = b
            return
        # if it is LASSO regression
        if method.lower() == "lasso":
            prev_b = b
            for curr_epoch in range(num_epochs):
                if curr_epoch > 0:
                    diff_b = np.abs(np.subtract(b, prev_b))
                    if np.amax(diff_b) < update_threshold:
                        self.coeff = b
                        return
                prev_b = b
                new_b = np.array([])
                # iterate through each component of b vector
                for i in range(len(b)):
                    bj = b[i]
                    # this component of b updated using soft threshold function
                    bj = bj - ((eta/col_x) * soft_thresh(matrix_x, b, i, lam, vector_y))
                    new_b = np.append(new_b, bj)
                # whole b vector updated
                b = new_b
            self.coeff = b
            return
        # if the input method is neither linear, lasso, or ridge, then an error message is printed and the program is
        # quit
        else:
            print("Error. The method argument should be either \"linear\", \"lasso\", or \"ridge\".")
            quit()

    """
    function to predict output with a fitted model given input feature values
    """
    def predict(self, matrix_x):
        matrix_x = np.asarray(matrix_x)
        err_flag = False
        # dimensions of feature matrix determined; if it does not match the expected dimensions, an error message is
        # printed and the program is immediately quit
        if len(matrix_x.shape) != 2:
            print("Error. The input X matrix should consist of 2 dimensions and constant row and column lengths.")
            quit()
        row_x, col_x = matrix_x.shape
        # if the first element of the X matrix is not of type np.ndarray it likely means that the input matrix did not
        # have constant row lengths; an error message is printed and error flag raised to true
        if not isinstance(matrix_x[0], np.ndarray):
            print("Error. The input X matrix should have constant row lengths.")
            err_flag = True
        # if the model hasn't been fitted yet, an error is printed and error flag raised to be true
        if len(self.coeff) == 0:
            err_flag = True
            print("Error. Please fit the model to data before predicting output for new data.")
        # if any of the elements of the input X matrix are non-numeric an error message is printed and error flag raised
        # to be true
        isnan = False
        temp_x = matrix_x.flatten()
        for i in temp_x:
            if not isinstance(i, numbers.Number):
                isnan = True
                break
        if isnan:
            print("Error. At least one of the values in the X matrix is non-numeric.")
            err_flag = True
        # if the number of features in the input X matrix does not match the length of the coefficient vector an error
        # message is printed and error flag raised to be true
        if col_x + 1 != len(self.coeff):
            err_flag = True
            print("Error. The number of rows in the X matrix should match the number of rows in the X matrix which"
                  "the model was trained on. The X matrix should have " + str(len(self.coeff)) + " rows.")
        # program quit if error flag raised to be true at any point
        if err_flag:
            quit()
        # column of ones prepended to X matrix
        add_col_x = np.ones(row_x)
        matrix_x = np.column_stack((matrix_x, add_col_x))
        col_x += 1
        # predicted output is X matrix multiplied with the coefficient matrix
        y_pred = np.matmul(matrix_x, self.coeff)
        # predicted values returned
        return y_pred


"""
main function to demonstrate class functionality on boston housing dataset
"""
if __name__ == '__main__':
    # boston dataset loaded with features and output separated
    boston_x, boston_y = load_boston(return_X_y=True)
    # features normalized
    boston_x = normalize(boston_x, axis=0)
    # data split into 70% training data and 30% testing data
    X_train, X_test, y_train, y_test = train_test_split(boston_x, boston_y, test_size=0.3)
    # linear, lasso, and ridge regression models instantiated
    lin_reg = LinearRegressionModel()
    lasso_reg = LinearRegressionModel()
    ridge_reg = LinearRegressionModel()
    # models fit to training data
    lin_reg.fit(X_train, y_train)
    # for lasso and ridge regularization, semi-optimal lambda values can be determined using a linear search
    lasso_lam = 0.1
    ridge_lam = 0.1
    lasso_reg.fit(X_train, y_train, "lasso", lasso_lam)
    ridge_reg.fit(X_train, y_train, "ridge", ridge_lam)
    # linear regression model predictions output and rmse and r2 calculated with actual output
    lin_reg_preds = lin_reg.predict(X_test)
    e_lin = np.asarray(y_test) - lin_reg_preds
    rmse_lin = np.sqrt(np.mean(e_lin**2))
    r2_lin = 1-(sum(e_lin**2)/sum((y_test-np.mean(y_test))**2))
    # lasso regression model predictions output and rmse and r2 calculated with actual output
    lasso_reg_preds = lasso_reg.predict(X_test)
    e_lasso = np.asarray(y_test) - lasso_reg_preds
    rmse_lasso = np.sqrt(np.mean(e_lasso**2))
    r2_lasso = 1-(sum(e_lasso**2)/sum((y_test-np.mean(y_test))**2))
    # ridge regression model predictions output and rmse and r2 calculated with actual output
    ridge_reg_preds = ridge_reg.predict(X_test)
    e_ridge = np.asarray(y_test) - ridge_reg_preds
    rmse_ridge = np.sqrt(np.mean(e_ridge**2))
    r2_ridge = 1-(sum(e_ridge**2)/sum((y_test-np.mean(y_test))**2))
    # accuracies output here
    print("Linear regression - rmse: " + str(rmse_lin) + ", r^2: " + str(r2_lin))
    print("LASSO regression (lambda = " + str(lasso_lam) + ") - rmse: " + str(rmse_lasso) + ", r^2: " + str(r2_lasso))
    print("Ridge regression (lambda = " + str(ridge_lam) + ") - rmse: " + str(rmse_ridge) + ", r^2: " + str(r2_ridge))
    # coefficient vectors for each model output here
    print("Linear regression model coefficient vector: \n", lin_reg.coeff)
    print("LASSO regression model coefficient vector: \n", lasso_reg.coeff)
    print("Ridge regression model coefficient vector: \n", ridge_reg.coeff)

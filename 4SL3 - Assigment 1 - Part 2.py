
######################
###### Part 1 ########
######################

import numpy as np
import matplotlib.pyplot as plt

N_train = 10
N_valid = 100
X_train = np.ones(10).reshape((-1, 1))
X_valid = np.ones(100).reshape((-1, 1))
X1_train = np.linspace(0., 1., 10)  # training set
X1_valid = np.linspace(0., 1., 100)  # validation set
np.random.seed(334)
t_train = (np.sin(4 * np.pi * X1_train) + 0.3 * np.random.randn(10)).reshape((-1, 1))
t_valid = (np.sin(4 * np.pi * X1_valid) + 0.3 * np.random.randn(100)).reshape((-1, 1))

x_train = np.linspace(0., 1., 10).reshape((-1, 1))
x_valid = np.linspace(0., 1., 100).reshape((-1, 1))

err_train_arr = []
err_valid_arr = []

# Declaring B vector to be used when M=9 and regularization needs to be used
B_train = np.diag([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
B_valid = np.diag([0., 1., 1., 1., 1., 1., 1., 1., 1., 1.])


# Function to compute Training Error
def getTrainingError(y_train_prediction, t_train, N):
    t_train = t_train.reshape((-1, 1))
    y_train_t = np.subtract(y_train_prediction, t_train)
    err_train_i = np.dot(y_train_t.T, y_train_t) / N

    return err_train_i[0][0]


# Function to compute Validation Error

def getValidationError(y_valid_prediction, t_valid, N):
    t_valid = t_valid.reshape((-1, 1))
    y_valid_t = np.subtract(y_valid_prediction, t_valid)
    err_valid_i = np.dot(y_valid_t.T, y_valid_t) / N

    return err_valid_i[0][0]


# Computes parameter vector w and y prediction based on obtained w vector
# Optional parameters hp_lambda (hyperparameter lambda) and B, which are used when M=9 and regularization performed
def computeYPrediction(X, t, hp_lambda=1, B=np.diag([1])):
    if (hp_lambda != 1):
        N = 10
        B *= 2 * hp_lambda
        A = np.linalg.inv(np.dot(X.T, X) + B * N / 2)
        c = np.dot(X.T, t)
        w = np.dot(A, c)

        y_prediction = np.dot(X, w)

        return y_prediction

    A = np.linalg.inv(np.dot(X.T, X))
    c = np.dot(X.T, t)
    w = np.dot(A, c)

    y_prediction = np.dot(X, w)

    return y_prediction


for i in range(1, 11):

    # Case when M=9, need to compute y prediction using hyperparameter lambda as 3rd argument in function call
    # Utilizes B diagonal vector as declared above
    if (i == 10):
        y_train_prediction = computeYPrediction(X_train, t_train, 5, B_train)
        y_valid_prediction = computeYPrediction(X_valid, t_valid, 5, B_valid)

    # Function call to compute parameter vector w and the prediction y based on w and store in y prediction variable
    else:
        y_train_prediction = computeYPrediction(X_train, t_train)
        y_valid_prediction = computeYPrediction(X_valid, t_valid)

    # Function call to to compute error based on y prediction
    err_train_i = getTrainingError(y_train_prediction, t_train, N_train)
    err_valid_i = getValidationError(y_valid_prediction, t_valid, N_valid)

    # Appends errors to lists which will be plotted against M
    err_train_arr.append(err_train_i)
    err_valid_arr.append(err_valid_i)

    plt.title("M={}".format(i - 1))
    plt.scatter(x_train, t_train, color="orange", label='Training Set')
    plt.plot(x_train, y_train_prediction, color='red', label='Training Predictions')
    plt.plot(x_valid, y_valid_prediction, color='green', label='Validation Predictions')
    plt.scatter(x_valid, t_valid, color="blue", label='Validation Set')
    plt.plot(x_valid, np.sin(4 * np.pi * x_valid), color="black", label='FTrue(x)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 0.75))
    plt.show()

    # Basis expansion, inserting feature vector to the power of i, where i represents M, as in Mth feature
    X_train = np.insert(X_train, i, X1_train ** i, axis=1)
    X_valid = np.insert(X_valid, i, X1_valid ** i, axis=1)

M = list(range(0, 10))
plt.title('M vs. Error')
plt.plot(M, err_train_arr, color='red', label='Training Error')
plt.plot(M, err_valid_arr, color='blue', label='Validation Error')
plt.legend(loc='upper right', bbox_to_anchor=(1.5, 0.75))
plt.show()


######################
###### Part 2 ########
######################


from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split


# function to compute parameter vector w and prediction based on vector w, and compute error of prediction
def getParameterVectorw(X, t):
    # returning parameter vector w as requested
    A = np.linalg.inv(np.dot(X.T, X))
    c = np.dot(X.T, t)
    w = np.dot(A, c)

    return w


def computewVectorAndError(X, t, X_test, t_test):
    # Computing w vector using linear algebra methods with matrices
    A = np.linalg.inv(np.dot(X.T, X))
    c = np.dot(X.T, t)
    w = np.dot(A, c)
    t_prediction = np.dot(X_test, w)

    t_subtract = np.subtract(t_prediction, t_test)
    err_train = np.dot(t_subtract.T, t_subtract) / (len(X_test))
    return err_train


# Function to plot error
def error_plotting(cv_error, test_error):
    k = np.linspace(1, 13, 13)
    plt.title('k vs. Error')
    plt.plot(k, cv_error, color='red', label='Cross Validation Error')
    plt.plot(k, test_error, color='blue', label='Test Error')
    plt.legend(loc='upper right', bbox_to_anchor=(1.5, 0.75))
    plt.legend()
    plt.show()


def computeCvError(XData=X, selectionSubset=[], latestMinErrorCol=None):
    err_cv = [0] * 13

    # loop through original boston X data
    for i in range(XData.shape[1]):
        err_fold = 0
        # access ith column of X data to store in S and reshape to correct matrix size
        S = XData[:, i].reshape(-1, 1)

        # After each iteration, column from X data will be deleted from X but then added to S
        if latestMinErrorCol is not None:
            # Adding latest minimum error column as defined below at end of this function
            S = np.append(S, latestMinErrorCol.reshape(-1, 1), axis=1)

        # K fold splitting of X data
        for train_index, test_index in kf.split(XData):
            S_train, S_test = S[train_index], S[test_index]
            t1_train, t1_test = t[train_index], t[test_index]

            # Skipping over iterations where index already exists in subset
            if i in selectionSubset:
                continue

            dummyColumn_train = np.ones(len(S_train))
            dummyColumn_test = np.ones(len(S_test))

            # adding dummy vector of 1's as first column
            S_train = np.insert(S_train, 0, dummyColumn_train, axis=1)
            S_test = np.insert(S_test, 0, dummyColumn_test, axis=1)

            # summing up all errors of this particular fold
            w = getParameterVectorw(S_train, t1_train)
            err_fold += computewVectorAndError(S_train, t1_train, S_test, t1_test)

        # add the average error by dividing sum calculated within fold by 10 to the ith index of err_test list
        err_cv[i] = err_fold / 10
        print('Feature:', i, '- Test Error = ', err_cv[i])
    lowest_err = min(i for i in err_cv if i > 0)
    minErrorIndex = err_cv.index(lowest_err)

    # Printing selection subset and parameter vector w for ease of understanding data as model is being trained
    print("Selection Subset: ", selectionSubset)
    print("Parameter Vector: ", w)

    # Finds column that has the lowest minimum of this iteration and returns it so it can be used again in this function
    latestMinErrorCol = XData[:, minErrorIndex]
    print("\n\n")
    return lowest_err, minErrorIndex, latestMinErrorCol, w


X, t = load_boston(return_X_y=True)
kf = KFold(n_splits=10, shuffle=True, random_state=334)
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=1 / 4, random_state=334)

# Definining empty lists which will be populated by function call to ComputeCvError which yields several outputs
selectionSubset = []
cv_error = []
test_error = []
latestMinErrorCol = None
wParameterVector = []
i = 0

basis = 1

# I change the basis value above, which then determines which model I am training
if basis == 1:
    print('no basis')
    title = "No Basis Expansion"
    Xtrain = X_train
    Xtest = X_test
elif basis == 2:
    print('x squared basis')
    title = "x squared - Basis Expansion"
    Xtrain = X_train ** 2
    Xtest = X_test ** 2
elif basis == 3:
    print('square root x basis')
    title = "square root of x - Basis Expansion"
    Xtrain = np.sqrt(X_train)
    Xtest = np.sqrt(X_test)

# Model is trained 13 times
while i < 13:
    # Save several outputs from function to be added to empty lists
    [error, errorIndex, latestMinErrorCol, w] = computeCvError(Xtrain, selectionSubset, latestMinErrorCol)
    selectionSubset.append(errorIndex)
    cv_error.append(error)
    wParameterVector.append(w)

    i += 1

# Find index of lowest error in cross validation list
minErrorIndex = cv_error.index(min(cv_error))
# Find best number of features in S selection subset
bestFeatures = selectionSubset[:minErrorIndex + 1]
print("Best Features:", len(bestFeatures), "\nCross-Validation Error:", min(bestFeatures))
print("selection Subset: ", selectionSubset)
print("Parameter Vector: ", w)

# Computing test error with train test split variables defined
for i in range(X.shape[1]):
    test_error.append(
        computewVectorAndError(Xtrain.T[selectionSubset[:i + 1]].T, t_train, Xtest.T[selectionSubset[:i + 1]].T,
                               t_test))

print("\nError -", title)
print("cv_error", cv_error)
# plotting function
error_plotting(cv_error, test_error)
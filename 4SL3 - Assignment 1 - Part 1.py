########## MASTER CODE #################
import numpy as np
import matplotlib.pyplot as plt

N_train = 10
N_valid = 100
X_train = np.ones(10).reshape((-1,1))
X_valid = np.ones(100).reshape((-1,1))
X1_train = np.linspace(0.,1.,10) # training set
X1_valid = np.linspace(0.,1.,100) # validation set
np.random.seed(334)
t_train = (np.sin(4*np.pi*X1_train) + 0.3 * np.random.randn(10)).reshape((-1,1))
t_valid = (np.sin(4*np.pi*X1_valid) + 0.3 * np.random.randn(100)).reshape((-1,1))

x_train = np.linspace(0.,1.,10).reshape((-1,1))
x_valid = np.linspace(0.,1.,100).reshape((-1,1))

err_train_arr = []
err_valid_arr = []

#Declaring B vector to be used when M=9 and regularization needs to be used
B_train = np.diag([0., 1., 1., 1., 1., 1., 1., 1., 1., 1. ])
B_valid = np.diag([0., 1., 1., 1., 1., 1., 1., 1., 1., 1. ])

#Function to compute Training Error
def getTrainingError(y_train_prediction,t_train,N):
    t_train = t_train.reshape((-1,1))
    y_train_t = np.subtract(y_train_prediction,t_train)
    err_train_i = np.dot(y_train_t.T,y_train_t)/N 
    
    return err_train_i[0][0]
#Function to compute Validation Error

def getValidationError(y_valid_prediction,t_valid,N):
    t_valid = t_valid.reshape((-1,1))
    y_valid_t = np.subtract(y_valid_prediction,t_valid)
    err_valid_i = np.dot(y_valid_t.T,y_valid_t)/N

    return err_valid_i[0][0]

#Computes parameter vector w and y prediction based on obtained w vector
#Optional parameters hp_lambda (hyperparameter lambda) and B, which are used when M=9 and regularization performed
def computeYPrediction(X,t,hp_lambda=1,B = np.diag([1])):
    
    if(hp_lambda != 1):
        N = 10
        B *= hp_lambda
        A = np.linalg.inv( np.dot(X.T,X) + B*N/2)
        c = np.dot(X.T,t)
        w = np.dot(A,c)
    
        y_prediction = np.dot(X,w)
    
        return y_prediction

    A = np.linalg.inv(np.dot(X.T,X))
    c = np.dot(X.T,t)
    w = np.dot(A,c)
       
    y_prediction = np.dot(X,w)
    
    return y_prediction

for i in range(1,11):
    #Case when M=9, need to compute y prediction using hyperparameter lambda as 3rd argument in function call
    #Utilizes B diagonal vector as declared above
    if(i==10):
        y_train_prediction = computeYPrediction(X_train,t_train,5,B_train)
        y_valid_prediction = computeYPrediction(X_valid,t_valid,5,B_valid)
        
    #Function call to compute parameter vector w and the prediction y based on w and store in y prediction variable
    y_train_prediction = computeYPrediction(X_train,t_train)
    y_valid_prediction = computeYPrediction(X_valid,t_valid)
        
    #Function call to to compute error based on y prediction 
    err_train_i = getTrainingError(y_train_prediction,t_train,N_train)
    err_valid_i = getValidationError(y_valid_prediction,t_valid,N_valid)

    #Appends errors to lists which will be plotted against M
    err_train_arr.append(err_train_i)
    err_valid_arr.append(err_valid_i)
    
    plt.title("M={}".format(i-1))
    plt.scatter(x_train,t_train,color="pink",label='Training Set')
    plt.plot(x_train,y_train_prediction,color='red',label='Training Predictions')
    plt.plot(x_valid,y_valid_prediction,color='green',label='Validation Predictions')
    plt.scatter(x_valid,t_valid,color="blue",label='Validation Set')
    plt.plot(x_valid,np.sin(4*np.pi*x_valid),color="black",label='FTrue(x)')
    plt.legend(loc='upper right',bbox_to_anchor=(1.5,0.75))
    plt.show()
    
    #Basis expansion, inserting feature vector to the power of i, where i represents M, as in Mth feature 
    X_train = np.insert(X_train,i,X1_train**i,axis=1)
    X_valid = np.insert(X_valid,i,X1_valid**i,axis=1)

M = list(range(0,10))
plt.title('M vs. Error')
plt.plot(M,err_train_arr,color='red',label='Training Error')
plt.plot(M,err_valid_arr,color='blue',label='Validation Error')
plt.legend(loc='upper right',bbox_to_anchor=(1.5,0.75))
plt.show()
    


'''
# Attributes or Parameters
    1. hyper:
        learning_rate,
        num_iterations,
        print_cost,
    2. parameter:
        w, shape(n_features, 1)
        dw,shape(n_features, 1)
        b, scalar number
        db,scalar number
        z, shape(1, n_samples)
        dz,shape(1, n_samples)
    3. dataset
        X, shape(n_features, n_samples)
        y, shape(1, n_samples)
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class classLR(object):
    def __init__(self, learning_rate=0.5,
                       num_iterations=2000,
                       print_cost=False):
        print('Yes, you did initiate classLR.')
        self.Hi()
        # initiate hyperparameters
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.print_cost = print_cost
    
    def Hi(self):
        print('Hello world.Edit by remote vim.')

    def fit(self, X_train, Y_train):
        # initialize parameters with zeros
        nx = X_train.shape[0]
        self.w, self.b = self.initialize_with_zeros(nx)

        parameters, grads, costs = self.optimize(X_train, Y_train)
        self.w, self.b = parameters['w'], parameters['b']

    def predict(self, X):
        '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))
        w = self.w.reshape(X.shape[0], 1)
        b = self.b
        # Compute vector "A" predicting the probabilities of a cat being present in the picture
        A = self.sigmoid(np.dot(w.T, X) + b)
        for i in range(A.shape[1]):
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            # In Python, A[i,j] = A[i][j]
            if A[0][i] <= 0.5:
                Y_prediction[0][i] = 0
            else:
                Y_prediction[0][i] = 1
        
        assert(Y_prediction.shape == (1, m))
        return Y_prediction

    def score(self, X, Y):
        Y_prediction = self.predict(X)
        # accuracy
        return 1 - np.mean(np.abs(Y_prediction - Y))

    def sigmoid(self,z):
        """
        Compute the sigmoid of z

        Arguments:
        z -- A scalar or numpy array of any size.
        Return:
        s -- sigmoid(z)

        >>> '%.10f' % sigmoid(0)
        '0.5000000000'
        >>> '%.10f' % sigmoid(3.0)
        '0.9525741268'
        >>> '%.10f' % sigmoid(2)
        '0.8807970780'

        """
        s = 1/(1+np.exp(-1*z))
        return s

    def initialize_with_zeros(self, dim):
        """
        This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
        
        Argument:
        dim -- size of the w vector we want (or number of parameters in this case)
        
        Returns:
        w -- initialized vector of shape (dim, 1)
        b -- initialized scalar (corresponds to the bias)
        """
        w = np.zeros((dim, 1))
        b = 0
        assert(w.shape == (dim, 1))
        assert(isinstance(b, float) or isinstance(b, int))
        return w, b

    def propagate(self, X, Y):
        """
        Implement the cost function and its gradient for the propagation explained above

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

        Return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
        """
        m = X.shape[1]
        # FORWARD PROPAGATION (FROM X TO COST)
        w = self.w
        b = self.b
        A = self.sigmoid(np.dot(w.T, X)+b)
        cost =  -1*(np.sum(Y*np.log(A)+(1-Y)*np.log(1-A), axis=1)/m)                                 # compute cost
        
        # BACKWARD PROPAGATION (TO FIND GRAD)
        dw = np.dot(X, A.T-Y.T)/m
        db = np.mean(A-Y)
        assert(dw.shape == w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())
        
        grads = {"dw": dw,
                 "db": db}
        
        return grads, cost

    def optimize(self, X, Y):
        """
        This function optimizes w and b by running a gradient descent algorithm
        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps
        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
        """
        w = self.w
        b = self.b
        num_iterations = self.num_iterations
        learning_rate = self.learning_rate

        costs = []
        
        for i in range(num_iterations):
            # Cost and gradient calculation
            grads, cost = self.propagate(X, Y)
            # Retrieve derivatives from grads
            dw = grads["dw"]
            db = grads["db"]
            # update rule 
            w = w - learning_rate*dw
            b = b - learning_rate*db
            
            # Record the costs
            if i % 100 == 0:
                costs.append(cost)
            
            # Print the cost every 100 training examples
            if self.print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" %(i, cost))

        params = {"w": w,
                  "b": b}
        grads = {"dw": dw,
                 "db": db}
        
        return params, grads, costs

    def readMNIST(self):
    	mnist = pd.read_csv('mnist_train.csv')
    	columns = ['label'] + ['pix'+str(i) for i in range(784)]
    	mnist.columns = columns

    	mnistAB = mnist[mnist.label<=1]
    	X = mnistAB.iloc[0:500, 1:].as_matrix()
    	Y = mnistAB.iloc[0:500, :1].as_matrix()
    	train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size=0.8, random_state=0)
    	return train_X, train_Y, test_X, test_Y

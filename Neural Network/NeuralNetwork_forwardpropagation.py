### Neural Network with Backpropagation Algorithm

#The code refers to the clasifying task of MNIST digits using 
#NN with one hidden layer
#
#This is easily extandable to multi-hidden layer NN!

### author: Dr. Marko Mitic
### year: 2015
### contact: miticm@gmail.com

 
import scipy.io
import scipy.sparse
import numpy as np

class NeuralNetwork:
    
    def __init__(self):
        mat=scipy.io.loadmat('ex4data1.mat')
        
        self.X=mat['X']
        self.y=mat['y']-1 #matlab notation :) #-1 because of difference in indexing between Matlab and Python
        self.m=np.shape(self.X)[0]
        self.n=np.shape(self.X)[1]
        self.C1=1 #regularization parameter
    
    def inittheta(self, num_input,num_hidden, num_labels):
        
        ## WITH THIS YOU GET 97.52% ACCURACY
        mat=scipy.io.loadmat('ex4weights')
        theta1=mat['Theta1']
        theta2=mat['Theta2']        
        
        #theta1=np.random.rand(num_input+1,num_hidden).T # +1 is for bias neuron
        #theta2=np.random.rand(num_hidden+1, num_labels).T # +1 is for bias neuron 
        #
        return theta1, theta2
        
    def sigmoid(self, z):
        
        s=1./(1+np.exp(-z))
        
        return s
    
    def forward (self, X, y, theta1, theta2):
        
        a1 = np.append(np.ones((self. m, 1)), X, axis=1) # adding bias term to input layer
        
        z2 = np.dot(theta1,a1.T)
        a2 =  self.sigmoid(z2)
        a2 = np.append(np.ones((self. m, 1)), a2.T, axis=1) # adding bias term to input layer
       
        z3 = np.dot(theta2, a2.T)
        a3=self.sigmoid(z3) #final hypothesis
        
        return a3
        

############################
def runNN():
    NN=NeuralNetwork()
    
    X=NN.X
    y=NN.y
    
    num_input=NN.n ##400
    num_hidden=25
    num_labels=10
    
    [theta1, theta2]= NN.inittheta(num_input, num_hidden, num_labels)        
    hy=NN.forward(X, y, theta1, theta2)
       
    pred = np.zeros((np.shape(X)[0], 1 ))
    pred[:,0] = np.argmax(hy,axis=0)
    accuracy=(np.mean((pred)==y))*100 

    return accuracy    
    
    
accuracy=runNN()
print accuracy


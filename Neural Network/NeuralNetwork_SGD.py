### Neural Network with Backpropagation Algorithm and mini-batch Stochastic Gradient Descent

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
        self.alpha=0.5 # learning rate
        self.epsilon=0.12 # range for weights 
        
    def inittheta(self, num_input,num_hidden, num_labels):     
        
        theta1=np.random.rand(num_input+1,num_hidden).T*2*self.epsilon-self.epsilon # +1 is for bias neuron
        theta2=np.random.rand(num_hidden+1, num_labels).T*2*self.epsilon-self.epsilon # +1 is for bias neuron 
        return theta1, theta2
        
    def sigmoid(self, z):
        
        s=1./(1+np.exp(-z))
        
        return s
    
    def forward (self, X, y, theta1, theta2):
        
        a1 = np.append(np.ones((np.shape(X)[0], 1)), X, axis=1) # adding bias term to input layer
        
        z2 = np.dot(theta1,a1.T)
        a2 =  self.sigmoid(z2)
        a2 = np.append(np.ones((np.shape(X)[0], 1)), a2.T, axis=1) # adding bias term to input layer
       
        z3 = np.dot(theta2, a2.T)
        a3=self.sigmoid(z3) #final hypothesis
        
        return a3, z2, a2, a1
        
    def groundtruth(self,y):
        labels=np.array(y).flatten()

        data   = np.ones(len(labels))
        indptr = np.arange(len(labels)+1)
          
        ground_truth = scipy.sparse.csr_matrix((data, labels, indptr))
        ground_truth = np.transpose(ground_truth.todense()) 
        
        return ground_truth
          
    def cost(self, y, hy, theta1, theta2, ground_truth):
               
        cost1=np.multiply(-ground_truth,(np.log(hy)))
        cost2=np.multiply((1-ground_truth),(np.log(1-hy)))
        
        reg=np.sum(np.power(theta1[:,1:],2))+np.sum(np.power(theta2[:,1:],2)) #regularization term (we are skipping the first term)
        total_cost=(1./np.shape(y)[0])*np.sum((cost1-cost2)) + self.C1*(1./(2*np.shape(y)[0]))*reg
        return total_cost       
    
    def sigmoidgradient(self, z):
        
        gz=self.sigmoid(z)
        gradz=np.multiply(gz,(1-gz))
        
        return gradz
    
    def backprop(self, a3, a2, a1, groundtruth, theta1, theta2, z2):
        
        del3=a3-groundtruth
        
        del2_1=np.dot(theta2.T[1:,:], del3) #exclude the first bias nod in theta2
        del2=np.multiply(del2_1, self.sigmoidgradient(z2))
        
        delta2=del3*a2        
        delta1=del2*a1
        
        Theta2Grad=(1./np.shape(groundtruth)[1])*delta2
        Theta1Grad=(1./np.shape(groundtruth)[1])*delta1
        
        reg1=self.C1*(1./np.shape(groundtruth)[1])*theta1[:,1:]
        reg2=self.C1*(1./np.shape(groundtruth)[1])*theta2[:,1:]
        
        Theta1Grad[:,1:]=Theta1Grad[:,1:]+reg1
        Theta2Grad[:,1:]=Theta2Grad[:,1:]+reg2
        
        theta1=theta1-self.alpha*Theta1Grad
        theta2=theta2-self.alpha*Theta2Grad

        
        return theta1, theta2

########################################################
def runNN():
    NN=NeuralNetwork()
    
    X=NN.X
    y=NN.y
    
    num_input=NN.n ##400
    num_hidden=25
    num_labels=10
    
    ###Definine mini-batch (typical: 256 samples)
    indicies=np.random.randint(np.shape(X)[0], size=256)    
    Xi=X[indicies,:]
    yi=y[indicies]
    
    [theta1, theta2] = NN.inittheta(num_input, num_hidden, num_labels)        

    gd=NN.groundtruth(yi) # ground_truth
    
    Maxiter=2000
    
    for i in range(0, Maxiter):
        
        ### Select 256 different samples in each iteration
        indicies=np.random.randint(np.shape(X)[0], size=256)    
        Xi=X[indicies,:]
        yi=y[indicies]        
        gd=NN.groundtruth(yi) # ground_truth

        [hy, z2, a2, a1] = NN.forward(Xi, yi, theta1, theta2) #hy==a3
    
        J = NN.cost(yi, hy, theta1, theta2, gd)   

        [theta1, theta2]=NN.backprop(hy, a2, a1, gd, theta1, theta2, z2)
        
        #print "Current cost is", J
        #print 'Parameters Theta0 and Theta1 are {} and {}, respectively' .format(theta1, theta2)
           
    [hy, z2, a2, a1] = NN.forward(X, y, theta1, theta2) #hy==a3
    pred = np.zeros((np.shape(X)[0], 1 ))
    pred[:,0] = np.argmax(hy,axis=0)
    
    accuracy=(np.mean((pred)==y))*100 
    
    return accuracy    
    
########################################################    
if __name__ == '__main__':
    accuracy=runNN()
    print accuracy

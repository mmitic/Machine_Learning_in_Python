#### Logistic regression with regularizaion optimized by BFGS algorithm

### author: Dr. Marko Mitic
### year: 2015
### contact: miticm@gmail.com

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


class LRREG:
    
    def __init__(self, data):
        self.x=data[:,0:-1]
        self.y=data[:,-1]
        self.m=len(data)
        self.C1=0.1 #regularization parameter
        
    def plotdata(self):       
        plot1=self.x[np.where(self.y==1)]
        plot2=self.x[np.where(self.y==0)]   
        ### 
        plt.figure()
        plt.plot(plot1[:,0], plot1[:,-1],'bo', label='Admitted')      
        plt.plot(plot2[:,0], plot2[:,-1],'ro', label='Not admitted')
        #plt.legend()             
        plt.show()
        
    def mapfeatures(self,x): #building polynomial features
        
        degree=6        
        X1=x[:,0]
        X2=x[:,1]
        
        X1.shape=(len(X1),1)
        X2.shape=(len(X2),1)
        
        m=np.shape(x[:,0])[0]
        newfeatures = np.ones(shape=(m,1))

        for i in range(1, degree + 1):
            for j in range(i + 1):
                r = (X1 ** (i - j)) * (X2 ** j)
                newfeatures = np.append(newfeatures, r, axis=1)
 
        return newfeatures
            
        
    def sigmoid(self,z):
        
        return 1./(1+np.exp(-z))
        
    def costandgradient(self, theta, x, y):
              
        z=np.dot(x,theta)
        hypothesis=self.sigmoid(z)
        
        ##compute cost
        firstterm=np.dot(y,np.log(hypothesis))
        secondterm=np.dot((1-y), np.log(1-hypothesis))
        
        J=(-1./self.m)*(firstterm+secondterm) + (self.C1/(2*self.m))*(sum(theta[1:len(theta)]**2)) ###second part is for regularization        
    
        ##compute gradient        
        grad0=(1./self.m)*np.dot((hypothesis-y),x[:,0])
        gradrest=(1./self.m)*np.dot((hypothesis-y),x[:,1:np.shape(x)[1]]) + (self.C1/self.m)*theta[1:len(theta)] ###second part is for regularization
    
        grad=np.append(grad0, gradrest)
        
        return [J, grad]
        
    def predict(self,theta,x):
        
        z=np.dot(x,theta)
        hypothesis=self.sigmoid(z)
        
        p=np.zeros(len(z))
        
        for i in range(0,len(z)):
            
            if hypothesis[i]>0.5:
                p[i]=1
            else:
                p[i]=0
                
        return p
        
    ###TODO:plot the nonlinear hypothesis

################################################################################
def runLR():
    data = np.loadtxt("data_LogisticR2.txt", comments="#", delimiter=",", unpack=False)
    
    lr=LRREG(data)
    lr.plotdata()
    
    X=lr.x
    y=lr.y
    
    X=lr.mapfeatures(X)
    initial_theta=np.zeros(X.shape[1])
    [J, grad]=lr.costandgradient(initial_theta, X, y)

    opt_solution = optimize.minimize(lr.costandgradient, initial_theta, args = (X,y), method = 'L-BFGS-B', jac = True, options = {'maxiter': 500})

    theta=opt_solution.x

    p=lr.predict(theta,X)

    accuracy=(np.mean(p==y))*100
    
    return theta, accuracy
        
[theta, accuracy]=runLR()
               
print 'Optimal parameters found by LR with BFGS algorithm are', theta
print 'Learning accuracy is', accuracy
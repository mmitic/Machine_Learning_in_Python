#### Logistic regression optimized by BFGS algorithm

### author: Dr. Marko Mitic
### year: 2015
### contact: miticm@gmail.com

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


class LinearREG:
    
    def __init__(self, data):
        self.x=data[:,0:-1]
        self.y=data[:,-1]
        self.m=len(data)
        
    def plotdata(self):       
        plot1=self.x[np.where(self.y==1)]
        plot2=self.x[np.where(self.y==0)]   
        ### 
        plt.figure()
        plt.plot(plot1[:,0], plot1[:,-1],'bo', label='Admitted')      
        plt.plot(plot2[:,0], plot2[:,-1],'ro', label='Not admitted')
        #plt.legend()             
        plt.show()
                
    def addones(self):
        X=np.append(np.ones([self.m,1]),self.x,1) 
        return X           
        
    def sigmoid(self,z):
        
        return 1./(1+np.exp(-z))
        
    def costandgradient(self, theta, x, y):
              
        z=np.dot(x,theta)
        hypothesis=self.sigmoid(z)
        
        ##compute cost
        firstterm=np.dot(y,np.log(hypothesis))
        secondterm=np.dot((1-y), np.log(1-hypothesis))
        
        J=(-1./self.m)*(firstterm+secondterm)         
    
        ##compute gradient        
        grad0=(1./self.m)*np.dot((hypothesis-y),x[:,0])
        gradrest=(1./self.m)*np.dot((hypothesis-y),x[:,1:np.shape(x)[1]])
    
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
        
    def plotresult(self,theta,x):
        
        plot1=self.x[np.where(self.y==1)]
        plot2=self.x[np.where(self.y==0)] 
        
        plotx=[min(x[:,1]), max(x[:,1])]
        ploty=[-1./theta[2]]*(np.dot(plotx,theta[1])+theta[0])  
        ### 
        plt.figure()
        plt.plot(plot1[:,0], plot1[:,-1],'bo', label='Admitted')      
        plt.plot(plot2[:,0], plot2[:,-1],'ro', label='Not admitted')
        
        plt.plot(plotx,ploty)
        plt.title('Model found by Logistic Regression')                 
        plt.show()
        

################################################################################
def runLR():
    data = np.loadtxt("data_LogisticR.txt", comments="#", delimiter=",", unpack=False)
    
    
    lr=LinearREG(data)
    lr.plotdata()
    
    X=lr.x
    X=lr.addones()
    y=lr.y
           
    initial_theta=np.zeros(X.shape[1])
    [J, grad]=lr.costandgradient(initial_theta, X, y)

    opt_solution = optimize.minimize(lr.costandgradient, initial_theta, args = (X,y), method = 'L-BFGS-B', jac = True, options = {'maxiter': 500})

    theta=opt_solution.x

    p=lr.predict(theta,X)

    accuracy=(np.mean(p==y))*100
    
    lr.plotresult(theta,X)
    
    return theta, accuracy
        
[theta, accuracy]=runLR()
    
print 'Optimal parameters found by BFGS are', theta
print 'Learning accuracy is', accuracy
### Multivariate batch (standard) gradient descent algorithm

### author: Dr. Marko Mitic
### year 2015
### contact: miticm@gmail.com

import numpy as np
import matplotlib.pyplot as plt

class MultivariateBatchGD: #Main class for Batch Multivariate Gradient Descent
      
    def __init__(self, data):
        ### data is given in several-column format, last column is output
         self.m=len(data)
         self.size=data.shape
         self.x=data
         self.y=self.x[:,-1]
      
    def featurescaling(self):
        ### feature scaling using mean normalization and std instead of range
        self.xscaled=0.*self.x[:,0:self.x.shape[1]-1]
        
        for i in range (0, self.x.shape[1]-1): # last column is the output
            self.xscaled[:,i]=(self.x[:,i]-np.mean(self.x[:,i]))/np.std(self.x[:,i])
        
        #self.yscaled=(self.y-np.mean(self.y))/np.std(self.y) ##this is if you want to scale the output!
        self.yscaled=self.y     
        return self.xscaled, self.yscaled            
    
    def thetas(self):
        #self.thetas=np.random.rand(np.shape(self.xscaled)[1]) #randomly set values for theta
        self.thetas=np.zeros(3)
        return self.thetas                            
                                                                
    def addones(self):
        self.xscaled=np.append(np.ones([self.m,1]),self.xscaled,1) 
        return self.xscaled
    
    def hypothesis(self):
        #self.hy=self.thetas*self.xscaled
        self.hy=np.dot(self.xscaled, self.thetas)
        return self.hy   
        
    def cost_function(self):
        "Return Cost function for hypothesis hy(of input x) and output y"
        #self.xtransp=self.xscaled.transpose()
        
        priv=np.dot(self.xscaled,(self.thetas.transpose()))
        self.J=(1./(2.*self.m))*sum((priv-self.y)**2) 
        
        #priv=np.dot(self.xscaled,(self.thetas.transpose()))
        #self.J=(1./(2.*self.m))*sum((priv-self.yscaled)**2)       
        return self.J
           
    def gradientdescent(self,alpha):
        "Perform gradient descent algorithm"
        self.a=alpha #leaarning rate
        temptheta=0.*np.arange(0.,np.shape(self.xscaled)[1])
        
        for i in range (0, self.x.shape[1]):    
            #temptheta[i]=self.thetas[i]-self.a*(1./self.m)*sum(((self.hy[:,i]-self.yscaled)*self.xscaled[:,i]))         
            temptheta[i]=self.thetas[i]-self.a*(1./self.m)*sum(((self.hy-self.yscaled)*self.xscaled[:,i]))         

        self.thetas=temptheta
        
        return self.thetas
        
    def plotcost(self,col):
        plt.figure()
        iters=np.arange(0,len(self.Jhistory))
        plt.plot(iters, self.Jhistory, col)
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost J')
        plt.show()

## END CLASS

################################################################################

#import data
data = np.loadtxt("data_MultiGD.txt", comments="#", delimiter=",", unpack=False)

gd=MultivariateBatchGD(data)

##initialize input and theta values
[gd.xscaled, gd.yscaled]=gd.featurescaling() ##dont scale output y! check Muktivariate BatchGD_class to see yscaled!
gd.xscaled=gd.addones()
gd.thetas=gd.thetas()

## starting hypothesis
gd.hy=gd.hypothesis()

#calculate initial gradient
gd.J=gd.cost_function()

max_iter=50
gd.Jhistory=np.zeros(max_iter)

for i in range (0,max_iter):
    
    gd.hy=gd.hypothesis()
    gd.J=gd.cost_function()
    gd.Jhistory[i]=gd.J   
    gd.thetas=gd.gradientdescent(0.08) ##learning rate==0.008    

    print "Current cost is", gd.J
    print 'Theta parameters are', gd.thetas
    
print 'Final Theta parameters', gd.thetas

gd.plotcost('b')

##END
### Univariate batch (standard) gradient descent algorithm

### author: Dr. Marko Mitic
### year 2015
### contact: miticm@gmail.com

import numpy as np
import matplotlib.pyplot as plt

class BatchGD: #Main class for Univariate Batch Gradient Descent
      
    def __init__(self, data):
        ### data is given in two-column format, first column is input, second column is output
        self.x=data[:,0]
        self.y=data[:,1]
        self.m=len(data[:,0])          
        
    def plotdata(self):        
        ### 
        plt.figure()
        plt.plot(self.x, self.y, 'ro')                
        plt.show()
    
    def hypothesis(self, theta0, theta1):
        "Return univariate hypothesis"
        self.t0=theta0
        self.t1=theta1
        self.hy=self.t0+self.t1*self.x ##current hypothesis
        return self.hy
        
    def cost_function(self):
        "Return Cost function for hypothesis hy(of input x) and output y"
        self.J=(1./(2.*self.m))*sum((self.hy-self.y)**2)
        return self.J
           
    def gradientdescent(self,alpha):
        "Perform gradient descent algorithm"
        self.a=alpha #leaarning rate
                     
        temptheta0=self.t0-self.a*(1./self.m)*sum((self.hy-self.y))
        temptheta1=self.t1-self.a*(1./self.m)*sum(((self.hy-self.y)*self.x))
        
        self.t0=temptheta0
        self.t1=temptheta1
        return self.t0, self.t1
        
    def plothyp(self,col,title):        
        "Plot current hypothesis"
        plt.figure()
        plt.plot(self.x, self.y, 'ro')
        plt.plot(self.x, self.hy, col)
        #plt.gca().set_xlim(left=0)
        #plt.gca().set_ylim(bottom=0)
        plt.title(title)
        plt.show()

## END CLASS

################################################################################

#import data
data = np.loadtxt("data_GD.txt", comments="#", delimiter=",", unpack=False)

gd=BatchGD(data)
## visualize the data

gd.plotdata()
##starting values for hupothesis hy=theta0+theta1*x
gd.t0=1.5
gd.t1=-0.7

#test and plot the starting hypothesis
gd.hy=gd.hypothesis(gd.t0,gd.t1)
gd.plothyp('b','Initial hypothesis')

#calculate initial gradient
gd.J=gd.cost_function()

#define error tolerance:
tol=5

while (gd.J>tol): # of course, you can do this with FOR also :)
                  #for i in range (0,1000):
    gd.hy=gd.hypothesis(gd.t0,gd.t1)
    gd.J=gd.cost_function()   
    [gd.t0,gd.t1]=gd.gradientdescent(0.002) ##learning rate==0.002    ## interesting results with 0.2 ;)

    print "Current cost is", gd.J
    print 'Parameters Theta0 and Theta1 are {} and {}, respectively' .format(gd.t0, gd.t1)
    
print 'Final parameters Theta0 and Theta1 found by GD are {} and {}, respectively' .format(gd.t0, gd.t1)
gd.plothyp('k', 'Final hypothesis')

##END
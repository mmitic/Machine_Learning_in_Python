### Univariate stochastic gradient descent algorithm

### author: Dr. Marko Mitic
### year 2015
### contact: miticm@gmail.com

import numpy as np
import matplotlib.pyplot as plt


class StochasticGD: #Main class for Univariate Stochastic Gradient Descent
      
    def __init__(self, data):
        ### data is given in two-column format, first column is input, second column is output
        self.x=data[:,0]
        self.y=data[:,1]
        self.m=len(data[:,0])          
   
   ### Just for stochastic GD
   
    def selectrandomsample(self,rnd):
        self.xrnd=self.x[rnd]
        self.yrnd=self.y[rnd]
        return self.xrnd, self.yrnd     
                                   
    def plotdata(self):
        ### 
        plt.figure()
        plt.plot(self.x, self.y, 'ro')                
        plt.show()
    
    def hypothesis(self, theta0, theta1):
        "Return univariate hypothesis"
        self.t0=theta0
        self.t1=theta1
        self.hy=self.t0+self.t1*self.xrnd ##current hypothesis
        return self.hy
        
    def cost_function(self):
        "Return Cost function for hypothesis hy(of input x) and output y"
        self.J=(1./(2.*self.m))*((self.hy-self.yrnd)**2)
        return self.J
           
    def gradientdescent(self,alpha):
        "Perform gradient descent algorithm"
        self.a=alpha #leaarning rate
                     
        temptheta0=self.t0-self.a*(1./self.m)*(self.hy-self.yrnd)
        temptheta1=self.t1-self.a*(1./self.m)*((self.hy-self.yrnd)*self.xrnd)
        
        self.t0=temptheta0
        self.t1=temptheta1
        return self.t0, self.t1
        
    def finalhypothesis(self, theta0, theta1):
        "Return univariate hypothesis"
        self.t0=theta0
        self.t1=theta1
        self.hyfin=self.t0+self.t1*self.x ##current hypothesis
        return self.hyfin
        
    def plothyp(self,col,tlt):
        "Plot current hypothesis"
        plt.figure()
        plt.plot(self.x, self.y, 'ro')
        plt.plot(self.x, self.hyfin, col)
        #plt.gca().set_xlim(left=0)
        #plt.gca().set_ylim(bottom=0)
        plt.title(tlt)
        plt.show()

## END CLASS

###############################################################################

### Univariate stochastic gradient descent algorithm

### author: Dr. Marko Mitic
### year: 2015
### contact: miticm@gmail.com

#import data
data = np.loadtxt("data_GD.txt", comments="#", delimiter=",", unpack=False)

gd=StochasticGD(data)
## visualize the data

gd.plotdata()
##starting values for hupothesis hy=theta0+theta1*x
gd.t0=1.5
gd.t1=-0.7

#test and plot the starting hypothesis
### Stochastic gradient descent
rnd=np.random.randint(0,gd.m)
[gd.xrnd,gd.yrnd]=gd.selectrandomsample(rnd)
gd.hy=gd.hypothesis(gd.t0,gd.t1)


#calculate initial gradient
gd.J=gd.cost_function()

#define error tolerance:
tol=5

#Adaptibile Learning rate
alpha=0.08

for i in range (0,100):
    
    rnd=np.random.randint(0,gd.m)
    [gd.xrnd,gd.yrnd]=gd.selectrandomsample(rnd)
    gd.hy=gd.hypothesis(gd.t0,gd.t1)
    gd.J=gd.cost_function()   
    [gd.t0,gd.t1]=gd.gradientdescent(alpha) ##learning rate==0.002    ## interesting results with 0.2 ;)
    
    print "Current cost is", gd.J
    print 'Parameters Theta0 and Theta1 are {} and {}, respectively' .format(gd.t0, gd.t1)
    
print 'Final parameters Theta0 and Theta1 found by GD are {} and {}, respectively' .format(gd.t0, gd.t1)

gd.hyfin=gd.finalhypothesis(gd.t0,gd.t1)

gd.plothyp('k','Final hypothesis')

##END
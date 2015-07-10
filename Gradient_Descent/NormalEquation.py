### Normal Equation for multivariate regression

### author: Dr. Marko Mitic
### year 2015
### contact: miticm@gmail.com

import numpy as np


class NormalEquation: #Main class for NormalEquation
      
    def __init__(self, data):
        ### data is given in several-column format, last column is output
         self.m=len(data)
         self.size=data.shape
         self.x=data[:,0:-1]
         self.y=data[:,-1]                                 
                                                                
    def addones(self):
        self.x=np.append(np.ones([self.m,1]),self.x,1) 
        return self.x
    
    def normaleq(self):
        pin=np.dot(self.x.transpose(),self.x)
        pin2=np.dot(self.x.transpose(),self.y)

        ## final result
        self.thetas=np.dot(np.linalg.pinv(pin), pin2)
        return self.thetas
        
## END CLASS

###############################################################################

#import data
data = np.loadtxt("data_MultiGD.txt", comments="#", delimiter=",", unpack=False)

gd=NormalEquation(data)

##initialize input
gd.x=gd.addones()

## Final result for thetas:
gd.thetas=gd.normaleq()
    
print 'Final Theta parameters', gd.thetas

##END
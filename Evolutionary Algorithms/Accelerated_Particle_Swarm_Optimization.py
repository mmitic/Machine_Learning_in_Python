### Evolutionary Algorithms
# 1. Accelerated Particle Swarm Optimization

### author: Dr. Marko Mitic
### year: 2015
### contact: miticm@gmail.com

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np


class APSO:
     
     def __init__(self, f, maxpop, numiter, bounds):
         
         # bounds =[Ub(x) Lb(x); Ub(y) Lb(y)]
         # define your function                  
             self.x=np.random.rand(maxpop)*(bounds[0,0]-bounds[0,1])+bounds[0,1]
             self.y=np.random.rand(maxpop)*(bounds[1,0]-bounds[1,1])+bounds[1,1]
             
         # algorithm specific parameters:   
             self.beta=0.5
             self.gamma=0.7
             
     def output(self,x,y,f):
             
             if f==1:                           
                self.z=np.power((1-x),2)+100*np.power((y-np.power(x,2)),2) #Rosenbrock
                
             if f==2:
                 self.z=-np.sin(x)*np.power((np.sin(np.power(x,2)/np.pi)),20)-np.sin(y)*np.power((np.sin(2*np.power(y,2)/np.pi)),20) #Michalewicz
                 
             return self.z
                         
     def plotfunc(self, bounds,f):       
                         
             xplot=np.arange(bounds[0,1], bounds[0,0], 0.1)
             yplot=np.arange(bounds[1,1], bounds[1,0], 0.1)
                          
             xplot,yplot=np.meshgrid(xplot, yplot)
             zplot=self.output(xplot, yplot, f)
             
             fig = plt.figure(figsize=plt.figaspect(0.5))
             ax = fig.gca(projection='3d')
             surf = ax.plot_surface(xplot, yplot, zplot, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
             plt.show()
      
     def rangeXY(self,x,y,bounds):
         
         for i in range(0, len(x)):
             if x[i] > bounds[0,0]:
                 x[i]=bounds[0,0]
             elif x[i] < bounds[0,1]:
                 x[i]=bounds[0,1] 
             elif y[i] > bounds[1,0]:
                 y[i]=bounds[1,0]                  
             elif y[i] < bounds[1,1]:
                 y[i]=bounds[1,1]                 
         return x, y
         
     def updateparticles(self,x,y,xo,yo,alpha,bounds):
         
         x=x*(1-self.beta)+xo*self.beta+alpha*np.random.randn(len(x))
         y=y*(1-self.beta)+yo*self.beta+alpha*np.random.randn(len(y))
         
         x,y =self.rangeXY(x,y,bounds)
         return x,y
     
########################################################################         
def runAPSO():
    
    f=2 # Choose a function to optimize
    
    ### Algorithm parameters:
    maxpop=20
    numiter=50
    
    ## Bounds for the function:
    bounds=np.array([[4,0],[4,0]])  # bounds =[Ub(x) Lb(x), Ub(y) Lb(y)]
    
    pso=APSO(f,maxpop,numiter, bounds)
    pso.plotfunc(bounds,f)
    
    x=pso.x
    y=pso.y
    
    x,y = pso.rangeXY(x,y,bounds)
    
    bestx=np.zeros(numiter)
    besty=np.zeros(numiter)
    bestz=np.zeros(numiter)
    
    for i in range (0, numiter):
        
        z = pso.output(x,y,f)
        idx=z.argmin() #index of minimum element in z
        xo=x[idx]
        yo=y[idx]
        zo=z[idx]
        
        alpha=np.power(pso.gamma, i+1)
        x,y = pso.updateparticles(x,y,xo,yo,alpha,bounds)
    
        bestx[i]=xo
        besty[i]=yo
        bestz[i]=zo
    
    print 'Minimum of the function is {}, in coordinates x = {} and y = {}' .format(zo,xo,yo)
 
########################################################################         
if __name__ == '__main__':
    runAPSO()
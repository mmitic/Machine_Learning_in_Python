#Firefly algorithm!
#Author: Marko Mitic
#Contact: miticm@gmail.com

from pylab import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

x=arange(0,4,4./50)
x=arange(-5,5,5./50)
y=x
ran=array([0,4,0,4])
ran=array([-5,5,-5,5])
x, y = np.meshgrid(x, y)
for i in range(len(x)):
    
    #f=-sin(x)*(sin((x**2)/np.pi))**20-sin(y)*(sin((2*y**2)/np.pi))**20
    f=(abs(x)+abs(y))*exp(-0.0625*(x**2+y**2))
    
fig = figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, f, rstride=1, cstride=1, cmap='hot')
xLabel = ax.set_xlabel('x')
yLabel = ax.set_ylabel('y')
zLabel = ax.set_zlabel('f(x,y)') #,linespacing=3.4
show()

def f_opt(x,y):
    '''function to optimize: 
    inputs: x,y
    output: f'''
    for i in range(len(x)):
        f=-sin(x)*(sin((x**2)/np.pi))**20-sin(y)*(sin((2*y**2)/np.pi))**20
        #f=(abs(x)+abs(y))*exp(-0.0625*(x**2+y**2))
    return f

def init_ffa(n,ran):
    '''initialize fireflies!!!: 
       n=number of swarm/ra, ran=range of x,y function values'''
      
    xn=zeros((n,1))
    yn=zeros((n,1))
    light=zeros((n,1))
    for i in range(n):
        
       xn[i]=rand(1)*(ran[1]-ran[0])+ran[0]
       yn[i]=rand(1)*(ran[3]-ran[2])+ran[2]
       
    return xn,yn,light

def findrange(xn,yn,rang):

    [t1,t2]=(xn<rang[0]).nonzero()
    xn[t1]=rang[0]
    [t1,t2]=(xn>rang[1]).nonzero()
    xn[t1]=rang[1]
    [t1,t2]=(yn<rang[2]).nonzero()
    yn[t1]=rang[2]
    [t1,t2]=(yn>rang[3]).nonzero()
    yn[t1]=rang[3]
    return xn,yn

def sortarray(zn):
    bb=sorted(zn)
    ind= sorted(range(len(zn)),key=lambda x:zn[x])
    return bb,ind
    
def move_ffa(xn,yn,lightn,xo,yo,lighto,alpha,gamma,rang):
    ni=len(yn)
    no=len(yo)
    for i in range(ni):
        for j in range(no):
            r=sqrt((xn[i]-xo[j])**2+(yn[i]-yo[j])**2)
            #print r
            if lightn[i]<lighto[j]:
                beta0=1
                beta=beta0*exp(-gamma*(r**2))
                xn[i]=xn[i]*(1-beta)+xo[j]*beta+alpha*(rand()-0.5)
                yn[i]=yn[i]*(1-beta)+yo[j]*beta+alpha*(rand()-0.5)
    
    [xn,yn]=findrange(xn,yn,rang)
    return xn,yn
    
#STARTING OPTIMIZATION:
#==============================================================================

alpha=0.2
gamma=1
ran=array([0,4,0,4])
ran=array([-5,5,-5,5])
n=12
maxiter=50
[xn,yn,lightn]=init_ffa(n,ran)

it=0
while it<maxiter:
    zn=f_opt(xn,yn)
    [lightn,ind]=sortarray(zn)
    xn=xn[ind]
    yn=yn[ind]
    xo=xn
    yo=yn
    lighto=lightn
    [xn,yn]=move_ffa(xn,yn,lightn,xo,yo,lighto,alpha,gamma,ran)
    it=it+1
    
bestx=xo
besty=yo
bestl=lighto

print bestx, besty, bestl
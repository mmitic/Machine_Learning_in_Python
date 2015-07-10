#Ant colony optimization! (S-ACO)
#Author: Marko Mitic
#Contact: miticm@gmail.com

from pylab import *
#import numpy as np
from mpl_toolkits.mplot3d import Axes3D

x=arange(-5,5,5./200)
y=x
rang=array([-5,5,-5,5])

#Rosenbrock(banana) function:
#x, y = np.meshgrid(x, y)
#
#for i in range(len(x)):
#    f=(x-1)**2+100*((y-x**2)**2)
#
#fig = figure()
#ax = Axes3D(fig)
#ax.plot_surface(x, y, f, rstride=1, cstride=1, cmap='hot')
#
#xLabel = ax.set_xlabel('x')
#yLabel = ax.set_ylabel('y')
#zLabel = ax.set_zlabel('f(x,y)') #,linespacing=3.4
#show()

def f_opt(x,y):
    '''function to optimize: 
    inputs: x,y
    output: f'''
    for i in range(len(x)):
        f=(x-1)**2+100*((y-x**2)**2)
    return f
    
def f_opt2(x,y):
    ''' function to optimize: single x,y pair'''
    f=(x-1)**2+100*((y-x**2)**2)
    return f
    
def initpheromone(x):
    '''initilize pheromone to uniform random values'''
    rr=len(x)
    tau=rand(rr)/2 #set up smaller random values by deviding them with 2)
    return tau

#def initantpose(nant,x,y):
#    f=f_opt(x,y)
#    ind=np.argmax(f)
#    startx=x[ind]
#    starty=y[ind]
#    #antx=ones((nant,1))*startx
#    #anty=ones((nant,1))*starty
#    #return antx,anty
#    return startx,starty

   
def split_list(alist, wanted_parts=1):
    '''split list into number of 'wanted parts' ''' 
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
             for i in range(wanted_parts) ]


def move_ant(nant,tau,al,x):
    '''move ant over the whole discretized x,y area'''

    whcrt=len(x) #whcrit (while criterion): length of x
    whcrt2=int(len(x)/10) #wanted_parts=10: devide value x,y values into 10 parts
 
    a=0
    b=whcrt2
           
    antk={} 
    k=0
    
    while k<nant: #for every ant in nant     
    
        ii=0 
        ant = {}
            
        while a<=whcrt-whcrt2: #while every layer (10 in total) is visited
                
            sumprob=sum(tau[a:b]**al) #sum all tau values in a layer
            ff=tau[a:b]/sumprob #ff=normalize tau values
            maxa=sum(ff)
                
            aco_rand=maxa*rand() #random number
            aco_action=0
            summ=ff[aco_action]
            
            #select an nod (action) in the next layer!:
            while (aco_rand>summ and aco_action<len(tau)/10):
                #print (aco_action)
                aco_action=aco_action+1
                summ=summ+ff[aco_action]
            
            #check next layer(after current is finished=while criterion satisfied)
            a=a+whcrt2
            b=a+whcrt2
                
            ant[ii] = aco_action #save path (all actions) in layers for one ant
            ii=ii+1
        
        #after one ant is finished, restart while criterion:
        a=0 
        b=whcrt2
        
        antk[k]=ant #save paths of all ants
        k=k+1
        
    return antk

#evaluate function
def evaluate(antk,vector,pp):
    '''evaluate function for each ant and for each selected value in layer '''
    layer6={}
    for p in range(len(antk)):
        iji=antk[p]
        layer6[p]=iji[6]
        for zz in range(len(iji)):
            prim=vector[zz]
            prim2=prim[iji[zz]]
            #since x&y are in the same range:
            #finding global best value!
            eva=f_opt2(prim2,prim2)
            if eva<pp:
                pp=eva
                ovo=prim2
    return ovo,layer6
 
#STARTING OPTIMIZATION:
#==============================================================================
t=0
nant=20
rho=0.02 #rho
al=0.2 #alpha  
maxiter=100

[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]= split_list(x,10)
vector=[x1,x2,x3,x4,x5,x6,x7,x8,x9,x10]

tau=initpheromone(x)

f=f_opt(x,y)
pp=max(f)

#expect to be layer 6, so i want the history of ant movement in this layer!
layer6_alliterations={} 

while t<maxiter:
    antk=move_ant(nant,tau,al,x)
    
    [ovo,layer6]=evaluate(antk,vector,pp)
    layer6_alliterations[t]=layer6
    itemindex = nonzero(x == ovo)
    tau=(1-rho)*tau
    tau[itemindex]=tau[itemindex]+tau[itemindex]/10
    
    t=t+1


print itemindex, tau[itemindex], x[itemindex], y[itemindex]

#Plotting results-ant movement:

a1={}
a2={}
a3={}
a4={}
a5={}

for i in range(12):
    j=int(i*8) #every 8th iteration!
    aan=layer6_alliterations[j]
    a1[i]=aan[0]
    a2[i]=aan[4]
    a3[i]=aan[7]
    a4[i]=aan[11] 
    a5[i]=aan[17]

#print ant selection (5 ants) in layer 6(expected layer) over 12 selected iteration (note that j=i*8!):
figure()
xlim(1, 12)
xlabel('iteration number') #, fontsize=18
ylabel('discretized value')
grid(True)
show()


aaa=a1.values()
priv=range(1,len(aaa)+1)
plot(priv, aaa, color="blue", linewidth=1.0, linestyle="-")
bbb=a2.values()
plot(priv, bbb, color="red", linewidth=1.0, linestyle="-")
ccc=a3.values()
plot(priv, ccc, color="black", linewidth=1.0, linestyle="-")
ddd=a4.values()
plot(priv, ddd, color="magenta", linewidth=1.0, linestyle="-")
eee=a5.values()
plot(priv, eee, color="green", linewidth=1.0, linestyle="-")
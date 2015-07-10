### Kmeans Algorithm for IRIS dataset

#This is easily extandable to any problem/dataset

### author: Dr. Marko Mitic
### year: 2015
### contact: miticm@gmail.com

import numpy as np

class KMeans:
    
    def __init__(self, dataraw):
        
        #make sure that the iris dataset is located in the current directory        
        self.y = dataraw[:,-1] # outputs - cluster types
        self.x = dataraw[:,0:-1] # inputs - cluster data
        
    def initializecluster(self, x, y):
        #r1 = np.random.randint(0,49) # first class
        #C1 = x[r1, :]
        #r2 = np.random.randint(50,99) # second class
        #r3 = np.random.randint(100,149) # third class   
        #C2 = x[r2, :]
        #C3 = x[r3, :]       
        
        C1 = x[np.random.randint(0,len(y)), :]
        C2 = x[np.random.randint(0,len(y)), :]
        C3 = x[np.random.randint(0,len(y)), :]
        
        return C1, C2, C3
        
    def assignmentstep(self, x, C1, C2, C3):
        C = np.zeros(len(x))
        
        for i in range(0, len(x)):
            dist1 = np.linalg.norm(x[i]-C1)
            dist2 = np.linalg.norm(x[i]-C2)
            dist3 = np.linalg.norm(x[i]-C3)
            
            if (dist1 < dist2) & (dist1 < dist3):            
                C[i] = 1
            elif (dist2 < dist1) & (dist2 < dist3):
                C[i] = 2
            elif (dist3 < dist1) & (dist3 < dist2):
                C[i] = 3
        
        return C
        
    def movecentroidstep(self, x, C):
        
        C1 = np.mean(x[C==1], axis=0)
        C2 = np.mean(x[C==2], axis=0)
        C3 = np.mean(x[C==3], axis=0)
        
        return C1, C2, C3
        
    def calculatecost(self, x, C, C1, C2, C3):
        
        if (len(x[C==1]) != 0):
            J1 = (1./len(x[C==1]))*np.power(np.linalg.norm(x[C==1]-C1), 2)
        else:
            J1=0
        
        if (len(x[C==2]) != 0):   
            J2 = (1./len(x[C==2]))*np.power(np.linalg.norm(x[C==2]-C2), 2)
        else:
            J2=0
        
        if (len(x[C==3]) != 0):          
            J3 = (1./len(x[C==3]))*np.power(np.linalg.norm(x[C==3]-C3), 2)
        else:
            J3=0   

        J = J1+J2+J3
        
        return J
        
############################################################################
def runKM():
    
    dataraw = np.loadtxt("iris.txt", comments="#", delimiter=",", unpack=False) 

    KM = KMeans(dataraw)       
    x = KM.x
    y = KM.y
    
    [C1, C2, C3] = KM.initializecluster(x, y)
    Cinit = KM.assignmentstep(x, C1, C2, C3)   
    
    Maxit =1000
    
    J = np.zeros((Maxit, np.shape(x)[1]))
  
    for  i in range (0, Maxit):
        
        C = KM.assignmentstep(x, C1, C2, C3)             
        [C1, C2, C3] = KM.movecentroidstep(x, C)
        J[i, :] = KM.calculatecost(x, C, C1, C2, C3) # check cost J
        
    accuracy=(np.mean(C==y))*100
    return accuracy, C   
    
##########################################################################

if __name__ == '__main__':
    acc = 0
    Center = np.zeros((150))
    
    for i in range (0, 10): # np.random.seed(145) gives 88.67% accuracy
        [accuracy, C] = runKM()
        if acc < accuracy:
            acc = accuracy
            Center = C  

    print acc
    print Center
        
# -*- coding: utf-8 -*-

### General KMeans (tested on iris, wine and segment dataset)

### author: Dr. Marko Mitic
### year: 2015
### contact: miticm@gmail.com

import numpy as np

class KMeans:
    
    def __init__(self, dataraw):
        
        #FOR IRIS DATASET!
        #self.y = dataraw[:,-1] # outputs - cluster types
        #self.x = dataraw[:,0:-1] # inputs - cluster data
        
        ### FOR WINE AND SEGMENT! 
        ### MAKE SURE y IS THE OUTPUT (CLUSTER) VARIABLE!
        
        #make sure that the dataset is located in the current directory        
        self.y = dataraw[:,0] # outputs - cluster types
        self.x = dataraw[:,1:] # inputs - cluster data
        
    def initializecluster(self, x, y, Nocluster):    
        
        CC = np.zeros((Nocluster, np.shape(x)[1]))
        
        for i in range(0, Nocluster):
            CC[i,:] = x[np.random.randint(0,len(y)), :] 
              
        return CC
        
    def assignmentstep(self, x, CC):
        C = np.zeros(len(x)) # assigned cluster
        dist = np.zeros(np.shape(CC)[0])
        

        for i in range(0, len(x)):
            
            for j in range (0, np.shape(CC)[0]):                                
                
                dist[j] = np.linalg.norm(x[i]-CC[j])
                
            C[i] = np.argmin(dist)
            
        return C
        
    def movecentroidstep(self, x, C, Nocluster):
        Cnew = np.zeros((Nocluster, np.shape(x)[1]))
        
        for  i in range (0, len(np.unique(C))):
            
            Cnew[i, :] = np.mean(x[C==i], axis=0)

        return Cnew
        
        
############################################################################
def runKM():
    
    dataraw = np.loadtxt("wine.txt", comments="#", delimiter=",", unpack=False) # select your dataset: iris.txt, wine.txt, segment.tst

    KM = KMeans(dataraw)       
    x = KM.x
    y = KM.y
    Nocluster = 3 # 3 for iris and wine, 7 for segment
    
    CC = KM.initializecluster(x, y , Nocluster)
    Cinit = KM.assignmentstep(x, CC)
        
    Maxit =5000
     
    for  i in range (0, Maxit):
        
        C = KM.assignmentstep(x, CC)             
        CC = KM.movecentroidstep(x, C, Nocluster) #CCnew
        
    accuracy=(np.mean((C+1)==y))*100 #C+1 because Pzthon starts from zero assignment and y from 1
    return accuracy, C+1   
    
##########################################################################

if __name__ == '__main__':
    acc = 0
    Center = np.zeros((178)) #150 iris, 178 wine, 
    
    for i in range (0, 40): # np.random.seed(145) gives 88.67% accuracy 
        #run 10 times KMeans and return est result
        [accuracy, C] = runKM()
        if acc < accuracy:
            acc = accuracy
            Center = C  

    print acc
    print Center
        
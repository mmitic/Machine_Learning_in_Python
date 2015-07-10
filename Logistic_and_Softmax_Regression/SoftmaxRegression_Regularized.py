import os
import numpy as np
import scipy.sparse
import scipy.optimize

class Softmax:
    
    def __init__(self):
        
        self.path='G:\MACHINE_LEARNING_ALGORITHMS\Logistic_and_Stochastic_Regression'### insert your path here!
        self.C1=0.0001 #weight decay (regularization parameter)      
                     
    def loadfile(self, im, path2):     
        
        if im:
            
            x = os.path.join(self.path, path2)
            f=open(x,'rb') #read binary
            
            magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)

            num_images = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
            num_rows = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
            num_cols = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
    
            images = np.fromfile(f, dtype=np.ubyte)
            images = images.reshape((num_images, num_rows * num_cols)).T           
            images = images.astype(np.float64) / 255 #normalize
            
            return images
            
        else: # labels
            
            y=os.path.join(self.path,path2)
            f=open(y,'rb') #read binary

            magic = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
            num_labels = np.fromfile(f, dtype=np.dtype('>i4'), count=1)
            labels = np.fromfile(f, dtype=np.ubyte)
            
            return labels
     
    def groundtruth(self, labels):
        
        labels=np.array(labels).flatten()
     
        #groundtruth=np.array(scipy.sparse.csr_matrix((np.ones(num_examples),(range(num_examples), labels-1))).todense())
        
        data   = np.ones(len(labels))
        indptr = np.arange(len(labels)+1)
          
        ground_truth = scipy.sparse.csr_matrix((data, labels, indptr))
        ground_truth = np.transpose(ground_truth.todense())       
        
        return ground_truth
    
    def costandgradient(self, theta, x, y):
        
        theta = theta.reshape(self.num_classes, self.input_size)
        
        z=np.dot(theta,x)
        ground=self.groundtruth(y)
        
        hy= np.exp(z) ## hypothesis        
        pr2= (np.sum(hy, axis=0))     
        prob=hy/pr2      
        
        firstterm=np.multiply(ground, np.log(prob))

        regul  = 0.5 * self.C1 * np.sum(theta * theta) #regularization_parameter
        
        J=(-1./self.num_samples)*(np.sum(firstterm)) + regul 
               
        firstterm=np.dot(ground-prob, x.T)
        grad=-(1./self.num_samples)*firstterm + self.C1 * theta
        grad=np.array(grad)
        grad=grad.flatten()
        
        return [J, grad]
        
    def predict(self, theta, x):
        
        theta = theta.reshape(self.num_classes, self.input_size)        
        z=np.dot(theta,x)
        
        hy= np.exp(z) ## hypothesis        
        pr2= (np.sum(hy, axis=0))     
        prob=hy/pr2
        
        pred = np.zeros((np.shape(x)[1], 1 ))
        pred[:,0] = np.argmax(prob,axis=0)
        
        return pred
        
##############################################################################
def runSR():    
    SR=Softmax()
    
    X=SR.loadfile(1,'train-images-idx3-ubyte') #Xtrain
    y=SR.loadfile(0,'train-labels-idx1-ubyte') #ytrain
    
    SR.num_samples=np.shape(X)[1]
    SR.input_size=np.shape(X)[0]
    SR.num_classes=long(len(np.unique(y)))
    
    initial_theta = 0.005 * np.asarray(np.random.normal(size = (SR.num_classes*SR.input_size, 1)))
    
    [J, grad]= SR.costandgradient(initial_theta,X,y)
    
    opt_solution = scipy.optimize.minimize(SR.costandgradient, initial_theta, args = (X,y), method = 'L-BFGS-B', jac = True, options = {'maxiter': 500})
    
    thetaopt=opt_solution.x
    
    Xtest=SR.loadfile(1,'t10k-images-idx3-ubyte')
    ytest=SR.loadfile(0,'t10k-labels-idx1-ubyte')
    
    pred=SR.predict(thetaopt, Xtest)
    correct=pred[:,0]==ytest
    accuracy = np.mean(correct)
    
    print 'Accuracy is', accuracy*100 
    
##############################################################################

runSR()
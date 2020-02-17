DEFAULT_BATCH_SIZE = 20

from numpy.linalg import norm as L2
from scipy.stats import norm as univariate_normal
from scipy.stats import multivariate_normal
from tqdm import tqdm_notebook
import numpy as np 

class KernelDensityEstimator:
    def __init__(self,kernel="multivariate_gaussian", bandwidth_estimator = "silverman",univariate_bandwidth = None):
        
        kernels = {"multivariate_gaussian":self.kernel_multivariate_gaussian,
                   "univariate_gaussian": self.kernel_univariate_gaussian}
        bandwidth_estimators = {"silverman":self.est_bandwidth_silverman,
                               "scott":self.est_bandwidth_scott,
                                "identity": self.est_bandwidth_identity}
        compatible_estimators = {"multivariate_gaussian":["silverman","scott","identity"],
                               "univariate":[]}
                    
            
        self.kernel =  kernels[kernel]

        # if multivariate gaussian kernel is chosen, choose an estimator
        if kernel=="multivariate_gaussian":
            self.bandwidth_estimator = bandwidth_estimators[bandwidth_estimator]
        
        # if choosing univariate kernel without bandwidth clarified, print out a warning
        elif kernel=="univariate_gaussian" and (not univariate_bandwidth):
            print("Please define your \"univariate_bandwidth\" parameters since the bandwidth cannot \
                    automatically estimated using univariate kernel yet")
        
        else:
            self.univariate_bandwidth = univariate_bandwidth
                    
        # Kernel choice
        self.kernel = kernels[kernel]

        # Bandwidth for estimating density
        self.bandwidth = None
        
        # Store data
        self.data = None
        
    def kernel_multivariate_gaussian(self,x):
        # Estimate density using multivariate gaussian kernel

        # Retrieve data
        data = self.data
        
        # Get dim of data
        d = data.shape[1]
        
        # Estimate bandwidth
        H = self.bandwidth_estimator()
        self.bandwidth = H

        # Calculate determinant of non zeros entry
        diag_H = np.diagonal(H).copy()
        diag_H[diag_H==0]=1
        det_H = np.prod(diag_H)

        # Multivariate normal density estimate of x
        var = multivariate_normal(mean=np.zeros(d), cov=H,allow_singular=True)
        density = np.expand_dims(var.pdf(x),1)
        return density
    
    def kernel_univariate_gaussian(self,x):
        # Estimate density using univariate gaussian kernel

        # Retrieve data
        data = self.data
        
        # Get dim of data
        d = data.shape[1]
        
        # Estimate bandwidth
        h = self.univariate_bandwidth
        # Calculate density
        density = univariate_normal.pdf(L2(x,axis=1)/h)/h
        
        return density

    def fit(self,X,y=None):
        
        self.data = X # Make a pointer to the data variable
        
        return self
        
    def eval(self,X,y,batch_size=DEFAULT_BATCH_SIZE):
        # Print out evaluation using MSE and CE
        MSE, CE = self.MSE_CE(X,y,batch_size=batch_size)
        print("Cross entropy",CE )
        print("Mean Square Error: ",MSE)
        return MSE,CE

    def MSE_CE(self,X,y,batch_size=DEFAULT_BATCH_SIZE):
        # Calculate mean square error and a binary cross entropy for a given H
        
        # Retrieve number of classes
        num_classes = len(np.unique(y))
        
        # Retrieve number of instances in X
        N = len(X)
        
        # Predict proba
        proba = self.predict_proba(X,batch_size=batch_size) + 1e-15 # to fix log(0)

        # Construct mean square error
        MSE = (proba.mean() - 1/num_classes)**2

        # Construct mean cross entropy
        CE = 1/N*np.sum(1/num_classes*np.log(proba) - (1-1/num_classes)*np.log(proba))

        return MSE, CE
      
    def est_bandwidth_scott(self):
        # Estimate bandwidth using scott's rule

        # Retrieve data
        data = self.data
        
        # Get number of samples
        n = data.shape[0]
        
        # Get dim of data
        d = data.shape[1]
        
        # Compute standard along each i-th variable
        std = np.std(data,axis=0) 
        
        # Construct the H diagonal bandwidth matrix with std along the diag
        H = (n**(-1/(d+4))*np.diag(std))**2
        
        return H
    def est_bandwidth_identity(self):
        # Generate an identity matrix of density for bandwidth

        # Retrieve data
        data = self.data
        
        # Get number of samples
        n = data.shape[0]
        
        # Get dim of data
        d = data.shape[1]

        # Construct the H bandwidth matrix
        H = np.identity(d)
        return H

    def est_bandwidth_silverman(self):
        # Estimate bandwidth using silverman's rule of thumbs

        # Retrieve data
        data = self.data
        
        # Get number of samples
        n = data.shape[0]
        
        # Get dim of data
        d = data.shape[1]
        
        # Compute standard along each i-th variable
        std = np.std(data,axis=0) 
        
        # Construct the H diagonal bandwidth matrix with std along the diag
        H = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.diag(std)
        return H
    
    def predict_proba(self,X,batch_size=10):
        # Predict proba for an input matrix X

        kernel_func = self.kernel

        # Retrieve data
        data = self.data
        
        # number of samples in data
        n_data = data.shape[0]
        # number of samples in input set
        n_X = X.shape[0]
        
        # Init the estimated probabilities list
        est_probs = np.empty(0)
        
        num_batches = np.ceil(n_X/batch_size)
        print("bs:",batch_size)
        for X_ in tqdm_notebook(np.array_split(X,num_batches)):

          # Add third dimension for broardcasting                          
          ## shape (1,dim,n_X)
          X_ = np.expand_dims(X,0).transpose((0,2,1)) 
          
          ## shape(n_data,dim,1)
          data_ = np.expand_dims(data,2) 
          
          # The difference of input set and data set pairwise (using broadcasting)
          
          ## shape (n_data,dim,n_X)
          delta = X_ - data_ 

          # Flatten the delta into matrix
          delta = delta.reshape(n_data*n_X,-1) # shape (n_data*n_X,dim)

          est_prob = kernel_func(delta) # (n_data*n_X,)

          # Calculate mean sum of probability for each sample
          est_prob = 1/n_data*est_prob.reshape(n_data,n_X).T.sum(axis=1)
          est_probs = np.concatenate((est_probs,est_prob))
            
        return est_probs

    def random_sample(self,scaling_factor):
        # Randomly generate a new sample from the dataset

        # Get H 
        H = self.bandwidth_estimator()*scaling_factor

        # Retrieve data
        data = self.data

        # Randomly pick a data point
        random_data = np.random.permutation(self.data)[0]
        
        # sample
        sample = np.random.multivariate_normal(mean=random_data,cov=H)

        # Print out predicted density for new sample
        print("Density new sample: " , self.predict_proba(np.expand_dims(sample,0))[0])

        return random_data,sample

    def predict(self,X,batch_size=DEFAULT_BATCH_SIZE):
        # Predict proba for a given X to belong to a dataset
        
        # if x is a vector (has 1 axis)
        if len(X.shape) == 1:
            # expand one more axis to represent a matrix
            X = np.expand_dims(X,0)
            
        proba = self.predict_proba(X,batch_size=batch_size)
                        
        return proba
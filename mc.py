import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.integrate import quad
from skopt import gp_minimize

from  numpy.linalg  import  inv,det, cholesky

from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF



def smc(f,x):
    T=x.shape[0]
    f_p=(1/T)*np.sum(f(x))
    variance=(1/T)*np.sum(f(x)**2-f_p**2)
    return f_p, variance

# I=quad(f1,0,1)(abs(f),0,1)

def ois(f,x,p,alpha):
    
    T=x.shape[0]
    q= lambda x: abs(f(x)) * p(x)/alpha
    f_p=smc(f,x)[0]
    
    f_hat_p=(1/T**2)*np.sum(f(x)*p(x)/q(x))
    
    variance= (1/T**2)*np.sum((f(x)**2)*p(x)**2/q(x)-f_p**2)
    
    return f_hat_p, variance

def bmc(X,y,p): 
    
    """
    
    Parameters:
    
    X (n,d)-array:  samples points  
    
    y  n-array:       evaluation of function to integrate on y
    
    Return 
    
    tuple containing the bmc integral expectation , the integral variance 
    """
    n, d= X.shape
    # fit a gaussain kernel gaussian processs regressor   with  the datas points to infer the function f
    kernel =RBF(length_scale=np.random.uniform(0,1,X.shape[1])) 
    gpr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X, y)
    # get the optimize hyperparameters w (length scale) i.e ( need to apply exponential because the default ouputs is in log_scale) 
    w=np.exp(gpr.kernel_.theta)
    A=np.diag(w)
    # get the distribution mean and covariance
    b, B=p.mean, p.cov
    # compute z for all data points
    w_0=(2*np.pi)**(-.5*d)*det(A)**(-.5)

    z= w_0*det( inv(A)@B + np.eye(d) )**(-0.5)*np.exp( -0.5*np.diag( (X-b)@inv(A+B)@(X-b).T ) )
    # compute the covariance matrice K with the optimized kernel
    K=gpr.kernel_.__call__(X, X, eval_gradient=False)
    # compute the integral expectation 
    E=z.T@inv(K)@y/n
    # compute the integral variance 
    V=(w_0*(1/np.sqrt(det(2*inv(A)@B + np.eye(d)))) - z.T@inv(K)@z)/n

    return E,V


class Gaussian_kernel:
    """
    
    anisotropic gaussian kernel 
    
    Attributes:
    
    length_scale: array  of shape the dimenions of hypperparameters
    
    
    Methods:
    
    __call__ : compute the gaussian covariance matrix  
              
              parameters: X (n,d)-array ,Y (m,d)-array 
    
    
        
    """
    def __init__(self,length_scale):
        self.length_scale=np.array(length_scale)
    def __call__(self,Y,X):
        K=[]
        n,d=Y.shape
        w=np.array(self.length_scale)
        for i in range(n):
            K.append(np.exp(-0.5*np.sum((Y[i,:]-X)**2/self.length_scale**2, axis=1)))
        return (2*np.pi)**(-.5*d)*det(np.diag(w))**(-.5)*np.array(K)
    
class Bmc:
    """
    Attributes
    
    kerne_t: (class object): the type of kernel  ie  Gaussian_kernel
    
    noise_level: (scalar>0):   noise variance to add on the vocvariance function diagonal coefficent
    
    kernel_ :           data point kernel instanciate, the covariance function
    
    L_:  cholesky decomposition of the covariance matrix
    
    alpha_: 
    
    log_marginal_likelihood_value_:  the value of the log_marginal likelihood
    
    X (n,d)-array, y (n-array) :     the data points
    
    n (scalar): number of samples in the fitted
    
    p:  (gaussian distribution):   the disribution of the data poit in the fitting
    
    d: (dimensions):   dimensions of the dat samples in the fitted
    
    integral_expectation (scalar):  output of the bmc estimate of the integral
    
    integral_variance (scalar): the ouput of the bmc integrale variance
    
    Methods
    
    log_marginal_likelihood:      compure the log marginal likelihood
    
    
    fit:          fit the data X,y p and optimize the kernel hyperparameters through a fixed gaussian optimiser  (gp_minimize)
    
    integrate: compute the integral expectation and variance; return tuple expectation ,variance
    
    
    """
    def __init__(self,kernel_type,noise_level):
        self.kernel_t=kernel_type
        self.noise_level=noise_level
        
    def log_marginal_likelihood(self,w):
        
        self.kernel_=self.kernel_t(w)
        K=self.kernel_.__call__(self.X,self.X)
        # L := cholesky(K + σ2nI)
        self.L_=cholesky(K+self.noise_level*np.eye(self.n))
        
        self.alpha_=inv(self.L_.T)@inv(self.L_)@self.y
        
        # log p(y|X) := −1/2y.Tα −sum log Lii −n/2log(2π) 
        self.log_marginal_likelihood_value_= -.5*self.y.T@self.alpha_ - np.sum(np.log(np.diag(self.L_))) -.5*self.n*np.log(2*np.pi)
        if self.d==1:
            return self.log_marginal_likelihood_value_[0][0]
        return self.log_marginal_likelihood_value_
    
    def fit(self, X,y,p):
        self.n,self.d=X.shape
        self.X,self.y=X,y
        self.p=p
        dimensions=[(1e-5,1e5) for i in range(self.d)]    # bound of kernels hyperparameters
        
        self.kernel_.lenght_scale=gp_minimize(self.log_marginal_likelihood,dimensions=dimensions,acq_func="EI",n_calls=15,n_random_starts=5,  
                                                noise=0.1**2,        
                                              random_state=1234)
        return self 
    
    def integrate(self):
        # get the optimize hyperparameters w (length scale) i.e ( need to apply exponential because the default ouputs is in log_scale) 
        w=self.kernel_.length_scale
        A=np.diag(w)
        # get the distribution mean and covariance
        b, B=self.p.mean, self.p.cov

        # compute z for all data points
        w_0=(2*np.pi)**(-.5*self.d)*det(A)**(-.5)
        z=w_0*det( inv(A)@B + np.eye(self.d) )**(-0.5)*np.exp( -0.5*np.diag( (self.X-b) @ inv(A+B) @ (self.X-b).T ) )
        # compute the covariance matrice K with the optimize kernel
        K=self.kernel_.__call__(self.X, self.X)

        # compute the integral expectation 
        self.integral_expectation= (z.T@inv(K)@self.y)/self.n
        # compute the integral variance 
        self.integral_variance=((w_0*det(2*inv(A)@B + np.eye(self.d))**(-.5))-z.T@inv(K)@z)/self.n
      
        return self.integral_expectation, self.integral_variance
    

    


        
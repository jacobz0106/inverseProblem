import numpy as np
import pandas as pd
import random
from scipy.optimize import fsolve

def Euclidean_distance(A, B):
        R2 = 0.0
        for i in range(len(A)):
            R2 += (A[i] - B[i])**2
        return np.sqrt(R2)

def Euclidean_distance_vector(B, A_train):
        return np.array([Euclidean_distance(B, A_i) for A_i in A_train ]) 


def Train(N):
    X = np.random.uniform(0, 5, N)
    Y = np.random.uniform(0, 2, N)
    Z = np.array([0]*N)
    Z[Y>=1] = 1
    df=pd.DataFrame(data={"X":X,"Y":Y,"Z":Z})
    return df
def Sample(n):
    X = np.random.uniform(0, 5, n)
    Y = np.random.uniform(0, 2, n)
    Z = np.array([0]*n)
    Z[Y>=1] = 1
    df=pd.DataFrame(data={"X":X,"Y":Y,"Z":Z})
    return df

def Generate_data(N):
    random.seed(10)  
    mu,sigma = 5, 1
    X = np.random.uniform(0, 10, N)
    Y = np.random.uniform(0, 10, N)
    Z = np.array([0]*N)
    #f=np.multiply(X,X)+np.multiply(Y,Y)
    f=np.multiply(Y,Y)+10* np.multiply(X,np.tanh(0.5*(X-5))) - 10*X
    I_1 = f <= np.quantile(f,0.25)
    I_2 =  np.logical_and( (f <= np.quantile(f,0.5)) , 
                      (f > np.quantile(f,0.25) ))
    I_3 =  np.logical_and( (f <= np.quantile(f,0.75)) , 
                      (f > np.quantile(f,0.5) ))
    Z[I_1] = 1
    Z[I_2] = 2
    Z[I_3] = 3
    Z[Z==1]
    ### adjust the number depends on function## Boundary
    B_1 = np.logical_and(f >= np.quantile(f,0.25)-5 ,f
                         <= np.quantile(f,0.25)+5)
    B_2 = np.logical_and( f >= np.quantile(f,0.5)-5 ,f
                         <= np.quantile(f,0.5)+5)
    B_3 = np.logical_and( f >= np.quantile(f,0.75)-5 ,f
                         <= np.quantile(f,0.75)+5)
    B = np.array([0]*N)
    B[B_1] = 1
    B[B_2] = 1
    B[B_3] = 1
    ##########built data frame
    df=pd.DataFrame(data={"X":X,"Y":Y,"Z":Z, "B":B})
    return df

def Generate_data_Curve(N, xlim=[0,10], ylim= [0,10]):
    X = np.random.uniform(xlim[0],xlim[1], N)
    Y = np.random.uniform(ylim[0], ylim[1], N)
    Z = np.array([0]*N)
    Z[Y > np.sin(X) + 5] = 1
    df=pd.DataFrame(data={"X":X,"Y":Y,"Z":Z})
    return df

def Generate_data_Circle(N, xlim=[0,10], ylim= [0,10]):
    X = np.random.uniform(xlim[0],xlim[1], N)
    Y = np.random.uniform(ylim[0], ylim[1], N)
    Z = np.array([0]*N)
    I = ((X-5)**2 + (Y-5)**2 >= 9)
    Z[I] = 1
    df=pd.DataFrame(data={"X":X,"Y":Y,"Z":Z})
    return df


def Generate_data_Curve_systematic(N, xlim=[0,10], ylim= [0,10]):
    X = np.tile(np.linspace(xlim[0],xlim[1], np.sqrt(N)), int(np.sqrt(N)) )
    Y = np.repeat(np.linspace(xlim[0],xlim[1], np.sqrt(N)), int(np.sqrt(N)) )
    Z = np.array([0]*N)
    Z[Y > np.sin(X) + 5] = 1
    df=pd.DataFrame(data={"X":X,"Y":Y,"Z":Z})
    return df

def Generate_data_Curve_half_circle(N):
    X = np.random.uniform(2,8,N)
    Y = np.repeat(0.0,len(X))
    Y = np.sqrt(9 - (X-5)**2) + 5 + np.random.normal(0,0.5,len(X))
    Z = np.repeat(1,len(X))
    X2 = np.random.uniform(3,7,N)
    Y2 = np.repeat(0.0,len(X2))
    Y2 = np.sqrt(4 - (X2-5)**2) + 5 + np.random.normal(0,0.5,len(X2))
    Z2 = np.repeat(0,len(X2))
    df=pd.DataFrame(data={"X":np.array([X,X2]).reshape(-1),"Y":np.array([Y,Y2]).reshape(-1),"Z":np.array([Z,Z2]).reshape(-1)})
    return df

#####Model
def function_y(lambda1, lambda2,lambda3,T,n):
    ls = np.zeros((n,2))
    ls[0] = (lambda3,1.0)
    t = T/n
    for i in range(1,n):
        def equations(vars):
            y1,y2 = vars
            eq1 = t*(lambda1 + y1**2*y2 - lambda2*y1 - y1) + ls[i-1][0] - y1
            eq2 = t*(lambda2*y1 - y1**2*y2) + ls[i-1][1] - y2
            return [eq1, eq2]
        ls[i] = fsolve(equations, ls[i-1])
    return ls

####Integral approximation
def integral(lambda1, lambda2,lambda3,T,n):
    #Riemann sum approxiamation left sided
    return (1/T)*np.sum(np.sum(function_y(lambda1, lambda2,lambda3,T,n),axis = 1)*T/n )


######

def createSystematic2D(n,X,Y):
    data = np.array([np.meshgrid(np.linspace(X[0],X[1],int(np.sqrt(n))),np.linspace(Y[0],Y[1],int(np.sqrt(n))))[0].reshape(-1),
               np.meshgrid(np.linspace(X[0],X[1],int(np.sqrt(n))),np.linspace(Y[0],Y[1],int(np.sqrt(n))))[1].reshape(-1)])
    return np.transpose(data)
    
def function_y(lambda1, lambda2,lambda3,T,n):
    """ 
    T: constant
    n: number of line spaces used to evaluate the intergral

    """
    ls = np.zeros((n,2))
    ls[0] = (lambda3,1.0)
    t = T/n
    # loop from t to T-t
    for i in range(1,n):        
        def equations(vars):
            y1,y2 = vars
            eq1 = t*(lambda1 + y1**2*y2 - lambda2*y1 - y1) + ls[i-1][0] - y1
            eq2 = t*(lambda2*y1 - y1**2*y2) + ls[i-1][1] - y2
            return [eq1, eq2]
        ls[i] = fsolve(equations, ls[i-1])
    return ls

####Integral approximation
def integral(lambda1, lambda2,lambda3,T,n):
    #Riemann sum approxiamation left sided
    return (1/T)*np.sum(np.sum(function_y(lambda1, lambda2,lambda3,T,n),axis = 1)*T/n )

##
def dy_dlambda1_t(lambda1, lambda2,lambda3,T,n):
    ls = np.zeros((n,2))
    ls[0] = (0,0)
    t = 0
    F = function_y(lambda1, lambda2,lambda3, T,n)
    for i in range(1,n):
        t += T/n
        Y = F[i]
        def equations(vars):
            y1,y2 = vars
            eq1=  T/n * (1 + 2*Y[0]*Y[1]*y1 + Y[0]**2*y2 - lambda2*y1 - y1)  + ls[i-1][0] - y1
            eq2=  T/n * (lambda2*y1 - 2*Y[0]*Y[1]*y1 - Y[0]**2*y2) + ls[i-1][1] - y2
            return [eq1, eq2]
        ls[i] = fsolve(equations, ls[i-1])
    return ls
def dy_dlambda2_t(lambda1, lambda2,lambda3,T,n):
    ls = np.zeros((n,2))
    ls[0] = (0,0)
    t = 0
    F = function_y(lambda1, lambda2,lambda3, T,n)
    for i in range(1,n):
        t += T/n
        Y = F[i]
        def equations(vars):
            y1,y2 = vars
            eq1=  T/n * (2*Y[0]*Y[1]*y1 + Y[0]**2*y2 - Y[0]-lambda2*y1 - y1)  + ls[i-1][0] - y1
            eq2=  T/n * (Y[0] + lambda2*y1 - 2*Y[0]*Y[1]*y1 - Y[0]**2*y2) + ls[i-1][1] - y2
            return [eq1, eq2]
        ls[i] = fsolve(equations, ls[i-1])
    return ls

def dQ_dlambda(lambda1, lambda2,lambda3 = 1.65,T = 5,n = 100):
    sum1 = (1/T)*np.sum(np.sum(dy_dlambda1_t(lambda1, lambda2,lambda3,T,n),axis = 1)*T/n )
    sum2 = (1/T)*np.sum(np.sum(dy_dlambda2_t(lambda1, lambda2,lambda3,T,n),axis = 1)*T/n )

    return np.array([sum1,sum2])






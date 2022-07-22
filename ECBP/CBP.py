import os
import sys
import numpy as np
from scipy.special import logsumexp
from collections import Counter
import random
import pdb
import math
import itertools
from itertools import product
from model_utility import Generate_data_Curve, Euclidean_distance, Euclidean_distance_vector, Generate_data_Curve_systematic, Generate_data_Curve_half_circle, dQ_dlambda
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from scipy import interpolate
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.optimize import fmin_cobyla,fsolve
from scipy.interpolate import BSpline, LinearNDInterpolator
from sklearn import datasets
from scipy import optimize 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import warnings
warnings.simplefilter('ignore', np.RankWarning)

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import pyearth
from skpp import ProjectionPursuitRegressor
from sklearn import cluster
from sklearn import svm

class LabelEncode(object):
  def __init__(self, L):
    self.L1 = np.unique(np.array(L))
    self.L2 = None
    self.dicForward= None
    self.dicBackward = None

  def fit(self,Y):
    self.L2 = np.unique(np.array(Y))
    self.dicForward= {self.L1[i]:self.L2[i] for i in range(len(self.L1))}
    self.dicBackward = {self.L2[i]:self.L1[i] for i in range(len(self.L1))}

  def transform(self, Y):
    Y = np.array(Y).reshape(-1)
    return np.array([self.dicBackward.get(i) for i in Y])

  def transform_back(self,Y):
    Y = np.array(Y).reshape(-1)
    return np.array([self.dicForward.get(i) for i in Y])


class CBP(object):
  '''
  Characteristic Boundary Points(CBP) for a training data set.
  Contains a list of (i,j) indexing points
  '''
  def __init__(self, A_train, C_train):

        self.count = 0
        #pairs of CBP points 
        self.points = []
        self.midpoints = []
        self.Euc_d = np.zeros(shape=(0,0))

        self.transformer = LabelEncode([0,1])
        self.transformer.fit(C_train)
        self.C_train = self.transformer.transform(C_train)

        self._train( A_train, self.C_train)
        self.midpoints = np.array(self.midpoints)

  def _train(self, A_train, C_train):
        '''
        Train the model, input nXd matrix A_train and 1Xn matrix labels
        '''
        self._Euc(A_train, C_train)
        for i in np.where(np.array(C_train) == 1)[0]:
          pointSet = np.where(np.array(C_train) == 0)[0]
          for j in pointSet:
            X_m = (np.array(A_train[i]) + np.array(A_train[j]) )/2
            for k in range(len(C_train)):
              if k == i or k ==j:
                continue 

              if self.Euc_d[i,j]**2 > self.Euc_d[i,k]**2 + self.Euc_d[j,k]**2:
                pointSet = pointSet[pointSet!=j]
                break
              else:
                #test whether p_k is inside pontSet:
                if k in pointSet:
                  #test whether P_j lies inside disk P_ik:
                  if self.Euc_d[i,k]**2 > self.Euc_d[i,j]**2 + self.Euc_d[j,k]**2:
                    pointSet = pointSet[pointSet!=k]
          #end
          if len(pointSet) >= 1:
            for m in pointSet:
              self.points.append([i,m])
              self.midpoints.append( (A_train[i] + A_train[m])/2)
              self.count += 1

  def _Euc(self, A_train, C_train):
        self.Euc_d = np.zeros(shape=(len(C_train),len(C_train)))
        self.Euc_d = np.array([Euclidean_distance_vector(A_i, A_train) for A_i in A_train])


""" Custom step-function """
class RandomDisplacementBounds(object):
    """random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
        Modified! (dropped acceptance-rejection sampling for a more specialized approach)
    """
    def __init__(self, xmin, xmax, stepsize=0.5):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds """
        min_step = np.maximum(self.xmin - x, -self.stepsize)
        max_step = np.minimum(self.xmax - x, self.stepsize)
        random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
        xnew = x + random_step

        return xnew



class CBPClassifier(object):
  '''
  Base classifier use Characteristic Boundary Points(CBP)
  '''
  def __init__(self,A_train, C_train):
    self.cbp = []
    self.avg = False
    self.upper = []
    self.lower = []
    self.kind = ''
    self.cond_min = np.min(A_train,axis = 0)
    self.cond_max = np.max(A_train,axis = 0)
    self.cond_min_upper = []
    self.cond_max_upper = []
    self.cond_min_lower = []
    self.cond_max_lower = []
    self.model_parameter = []
    self.Point = []
    self.tck_upper = []
    self.tck_lower = [] 
    self.d =[]
    self.transformer = LabelEncode([0,1])
    self.transformer.fit(C_train)
    self.C_train = self.transformer.transform(C_train)

    self._train(A_train, self.C_train)
    self.A_train = A_train

  def fit(self, average = False, kind = 'regression', args = []):
    '''
    regression: with parameter(degree) 
    gaussian:
    parameter must be array like, [upper, lower]
    PPR:
    '''
    self.avg = average
    self.model_parameter = args
    self.kind = kind
    self.d = len(self.A_train[0])
    if self.kind == 'gaussian':
      kernel = C(3, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
      dy = np.repeat(self.model_parameter[0],len(self.upper[:,-1]))
      gp = GaussianProcessRegressor(kernel=kernel,alpha = dy**2, n_restarts_optimizer=9)
      gp.fit(np.atleast_2d(self.upper[:,:-1]).reshape(-1, 1),  self.upper[:,-1].reshape(-1, 1))
      self.model_upper = gp
      dyy = np.repeat(self.model_parameter[1],len(self.lower[:,-1]))
      gl = GaussianProcessRegressor(kernel=kernel,alpha = dyy**2, n_restarts_optimizer=9)
      gl.fit(np.atleast_2d(self.lower[:,:-1]).reshape(-1, 1), self.lower[:,-1].reshape(-1, 1))
      self.model_lower=gl
      return

    elif self.kind =='1dSpline':
      filtered0 = np.copy(self.upper)
      filtered1 = np.copy(self.lower)
      self.model_upper =interpolate.interp1d( np.array(filtered0[:,:-1]).reshape(-1),  np.array(filtered0[:,-1]).reshape(-1),fill_value = "extrapolate", kind = args[0])
      self.model_lower =interpolate.interp1d( np.array(filtered1[:,:-1]).reshape(-1),  np.array(filtered1[:,-1]).reshape(-1),fill_value = "extrapolate", kind = args[0])

    elif self.kind =='PPR':
      self.model_upper = ProjectionPursuitRegressor(r = self.model_parameter[0],degree = self.model_parameter[1])
      self.model_upper.fit(np.array(self.upper[:,:-1]), self.upper[:,-1] )
      self.model_lower = ProjectionPursuitRegressor(r = self.model_parameter[2],degree = self.model_parameter[3])
      self.model_lower.fit(np.array(self.lower[:,:-1]), self.lower[:,-1] )

    elif self.kind =='MARS':
      self.model_upper = pyearth.Earth(max_terms=self.model_parameter[0],max_degree=self.model_parameter[1], penalty=self.model_parameter[2])
      self.model_upper.fit(np.array(self.upper[:,:-1]), self.upper[:,-1] )
      self.model_lower = pyearth.Earth(max_terms=self.model_parameter[3],max_degree=self.model_parameter[4], penalty=self.model_parameter[5])
      self.model_lower.fit(np.array(self.lower[:,:-1]), self.lower[:,-1] )

    elif self.kind =='spline':
      #upper
      x = self.upper[:,0]
      y = self.upper[:,1]
      tck_upper,u = interpolate.splprep([x,y],k=self.model_parameter[0],s=self.model_parameter[1])
      self.tck_upper = tck_upper
      #lower
      x = self.lower[:,0]
      y = self.lower[:,1]
      tck_lower,u = interpolate.splprep([x,y],k=self.model_parameter[2],s=self.model_parameter[3])
      self.tck_lower = tck_lower


  def _train(self, A_train, C_train):
    self.cbp = CBP(A_train, C_train)
    upper = np.unique(np.array(self.cbp.points)[:,0])
    self.upper = np.zeros(shape = (len(upper),len(A_train[0])))
    for i in range(len(upper)):
      self.upper[i] = A_train[upper[i]]
    
    lower = np.unique(np.array(self.cbp.points)[:,1])
    self.lower = np.zeros(shape = (len(lower),len(A_train[0])))
    for i in range(len(lower)):
      self.lower[i] = A_train[lower[i]]

    self.cond_min_upper = np.min(self.upper,axis = 0)
    self.cond_max_upper = np.max(self.upper,axis = 0)
    self.cond_min_lower = np.min(self.lower,axis = 0)
    self.cond_max_lower = np.max(self.lower,axis = 0)


  #possible regression methods.
  def regression_upper(self,x):
    return self.model_upper(x)
  def regression_lower(self,x):
    return self.model_lower(x)

  def gaussian_upper(self,x):
    return self.model_upper.predict(np.array(x).reshape(-1, 1))[0][0]
  def gaussian_lower(self,x):
    return self.model_lower.predict(np.array(x).reshape(-1, 1))[0][0]

  def spline1d_upper(self,x):
    return self.model_upper(x)
  def spline1d_lower(self,x):
    return self.model_lower(x)

  def PPR_upper(self,x):
    x = np.array(x).reshape(-1 ,self.d - 1)
    return self.model_upper.predict(x)
  def PPR_lower(self,x):
    x = np.array(x).reshape(-1 ,self.d - 1)
    return self.model_lower.predict(x)

  def MARS_upper(self,x):
    return self.model_upper.predict(x)
  def MARS_lower(self,x):
    return self.model_lower.predict(x)

  def Gradient_lower(self,x):
    I = Euclidean_distance_vector(x,self.lower[:,0:2])
    k = self.model_parameter[0]
    index = I.argsort()[0:k]
    #define weights
    approximates = np.zeros(k)
    orthDistance = np.zeros(k)
    projecton = np.zeros(k)
    for i in range(len(index)):
      A = self.lower[index[i]][0:2]
      gradient = dQ_dlambda(A[0], A[1])
      directionVector = x - A
      orthProj = np.sum(directionVector*gradient)/np.linalg.norm(gradient)
      distance = Euclidean_distance(x, A)
      projecton[i] = np.abs(orthProj)
      if distance == 0.0:
        orthDistance[i]= 9999
      else:
        orthDistance[i] = distance#np.sqrt(distance**2 - orthProj**2)
      #approximates[i] = self.lower[index[i]][2] + gradient*(x - A)
    weights = 1/orthDistance/np.sum(1/orthDistance)
    return np.sum(weights*projecton)

  def Gradient_upper(self,x):
    I = Euclidean_distance_vector(x,self.upper[:,0:2])
    k = self.model_parameter[0]
    index = I.argsort()[0:k]
    #define weights
    approximates = np.zeros(k)
    orthDistance = np.zeros(k)
    projecton = np.zeros(k)
    for i in range(len(index)):
      A = self.upper[index[i]][0:2]
      gradient = dQ_dlambda(A[0], A[1])
      directionVector = x - A
      orthProj = np.sum(directionVector*gradient)/np.linalg.norm(gradient)
      distance = Euclidean_distance(x, A)
      projecton[i] = np.abs(orthProj)
      if distance == 0.0:
        orthDistance[i] = 9999
      else:
        orthDistance[i] = distance#np.sqrt(distance**2 - orthProj**2)
      #approximates[i] = self.upper[index[i]][2] + gradient*(x - A)
    weights = orthDistance/np.sum(orthDistance)
    return np.sum(weights*projecton)

  def Spline_upper(self,x):
    return interpolate.splev(x,self.tck_upper)[1]

  def Spline_lower(self,x):
    return interpolate.splev(x,self.tck_lower)[1]



  # fit function
  def fit_upper(self,x):
    x = np.array(x).reshape(-1 ,self.d - 1)
    if self.kind == 'regression':
      return self.regression_upper(x)
    elif self.kind == 'gaussian':
      return self.gaussian_upper(x)
    elif self.kind == '1dSpline':
      return self.spline1d_upper(x)
    elif self.kind == 'PPR':
      return self.PPR_upper(x)
    elif self.kind == 'MARS':
      return self.MARS_upper(x)
    elif self.kind =='gradient':
      return Gradient_upper(x)
    elif self.kind == 'Spline':
      return self.Spline_upper(x)

  def fit_lower(self,x):
    x = np.array(x).reshape(-1 ,self.d - 1)
    if self.kind == 'regression':
      return self.regression_lower(x)
    elif self.kind == 'gaussian':
      return self.gaussian_lower(x)
    elif self.kind == '1dSpline':
      return self.spline1d_lower(x)
    elif self.kind == 'PPR':
      return self.PPR_lower(x)
    elif self.kind == 'MARS':
      return self.MARS_lower(x)
    elif self.kind =='gradient':
      return Gradient_lower(x)
    elif self.kind == 'Spline':
      return self.Spline_lower(x)



  def Multi_inter_lower(self,x):
    f = LinearNDInterpolator(list(zip()),self.lower)




  def Objective_lower(self,x,p):
    A = np.zeros(self.d)
    A[:-1] = np.array(x)
    A[-1] = self.fit_lower(x)
    return Euclidean_distance(A, p)

  def Objective_upper(self,x,p):
    A = np.zeros(self.d)
    A[:-1] = np.array(x)
    A[-1] = self.fit_upper(x)
    return Euclidean_distance(A, p)

  def const_upper(self,x):
    return (x-self.cond_min_upper)*(self.cond_max_upper-x) 

  def const_lower(self,x):
    return (x-self.cond_min_lower)*(self.cond_max_lower-x) 

  def predictSingle(self, x):
    x = np.array(x).reshape(-1,self.d)
    if self.kind =='gradient':
      return self.Gradient_upper(x)<= self.Gradient_lower(x)
    elif self.avg == True:
      x = np.array(x).reshape(-1,1)
      boundary = (self.fit_lower(x[:-1]) + self.fit_upper(x[:-1]))/2
      if x[-1] > boundary:
          return 1
      else:
          return 0
    x = np.array([x]).reshape(-1)
    if x[-1] > self.fit_upper(x[:-1]):
        return 1
    elif x[-1] < self.fit_lower(x[:-1]):
        return 0
    bounds_upper = optimize.Bounds(self.cond_min_upper[:-1], self.cond_max_upper[:-1])
    bounds_lower = optimize.Bounds(self.cond_min_lower[:-1], self.cond_max_lower[:-1])
    bounded_step_upper = RandomDisplacementBounds(self.cond_min_upper[:-1], self.cond_max_upper[:-1], stepsize = 1.5)
    bounded_step_lower = RandomDisplacementBounds(self.cond_min_lower[:-1], self.cond_max_lower[:-1], stepsize = 1.5)
    """ Custom optimizer """
    minimizer_kwargs_upper = {"method":"L-BFGS-B",  "bounds": bounds_upper, "tol":0.1, 'args':x}
    minimizer_kwargs_lower = {"method":"L-BFGS-B",  "bounds": bounds_lower, "tol":0.1, 'args':x}

    """ Solve with bounds """
    with np.errstate(divide='ignore',invalid='ignore'):
      x_upper = optimize.basinhopping(self.Objective_upper, x[:-1], minimizer_kwargs=minimizer_kwargs_upper, niter=200, take_step=bounded_step_upper ).x
      x_lower = optimize.basinhopping(self.Objective_lower, x[:-1], minimizer_kwargs=minimizer_kwargs_lower, niter=200, take_step=bounded_step_lower ).x



    # x_upper = fmin_cobyla(self.Objective_upper,x[:-1], self.const_upper, consargs = (), args = ([x]))
    # x_lower = fmin_cobyla(self.Objective_lower, x[:-1], self.const_lower, consargs = (), args = ([x]))

    P_upper = np.zeros(self.d)
    P_upper[:-1] = np.array(x_upper)
    P_upper[-1] = self.fit_upper(x_upper)
    P_lower = np.zeros(self.d)
    P_lower[:-1] = np.array(x_lower)
    P_lower[-1] = self.fit_upper(x_lower)
    d_upper = Euclidean_distance(P_upper, x)
    d_lower = Euclidean_distance(P_lower, x)
    return int(d_upper <= d_lower)


  def predict(self, x):
    predict = np.vectorize(self.predictSingle, signature = '(n)->()')(np.array(x))
    return self.transformer.transform_back(predict)

# referenced method: linear ensemble 
class refrenced_method(object):
  '''
  C_train must have label{1, -1}
  '''
  def __init__(self,A_train, C_train, alpha, constLambda):
    self.cbp = []
    self.alpha = alpha
    self.constLambda = constLambda
    self.weights = []
    self.A_train = A_train

    self.transformer = LabelEncode([-1,1])
    self.transformer.fit(C_train)
    self.C_train = self.transformer.transform(C_train)

    self._train(A_train, self.C_train)

  def _train(self, A_train, C_train):
    self.cbp = CBP(A_train, C_train)
    self.weights = np.zeros(self.cbp.count)
    A = np.zeros(shape = (len(A_train), self.cbp.count))
    for i in range(len(A_train)):
      A[i,:] = self.baseClassifier(A_train[i])
    initialWeights = np.repeat(1/self.cbp.count, self.cbp.count)
    self.weights = np.matmul(self.constLambda**2*initialWeights + np.matmul(np.transpose(A), self.C_train),
                             np.linalg.inv(np.matmul(np.transpose(A), A) + np.identity(self.cbp.count)*self.constLambda**2)
                    )

  def baseClassifier(self,x):
    classifiers = np.zeros(self.cbp.count)
    for i in range(self.cbp.count):
      midPoint = self.cbp.midpoints[i]
      upperPoint = self.A_train[self.cbp.points[i][0]]
      lowerPoint = self.A_train[self.cbp.points[i][1]]
      disc= sum((x - midPoint)*(upperPoint - lowerPoint))
      if disc >= 0:
        classifiers[i] = 1
      else:
        classifiers[i] = -1
    return classifiers

  def ensemble(self,x):
    func = np.sum(self.weights*self.baseClassifier(x)) - self.alpha
    if func >= 0:
      return 1
    else:
      return -1

  def predict(self, x):
    ensembleVec = np.vectorize(self.ensemble, signature = '(n)->()')
    predict = ensembleVec(x)
    return self.transformer.transform_back(predict)


class GPSVM(object):
  def __init__(self,A_train, C_train, args):
    self.cbp = []
    self.clusterNum = args[0]
    self.clusterCentroids = []
    self.ensembleNum = args[1]
    self.kmeans = []
    self.SVM = []
    self.clusterLabel = []
    #[0,1] -> [1, -1]

    self.transformer = LabelEncode([-1,1])
    self.transformer.fit(C_train)
    self.C_train = self.transformer.transform(C_train)

    self._train(A_train, self.C_train)

  def _train(self, A_train, C_train):
    self.cbp = CBP(A_train, C_train)
    if self.cbp.count < self.clusterNum:
      self.clusterNum = self.cbp.count
    if self.ensembleNum > self.clusterNum:
      self.ensembleNum = self.clusterNum

    self.kmeans = cluster.KMeans(n_clusters=self.clusterNum, random_state=0).fit(np.array(self.cbp.midpoints))
    self.clusterCentroids = self.kmeans.cluster_centers_
    self.clusterLabel= np.unique(self.kmeans.labels_)
    for i in range(self.clusterNum):
      midpoints_subset_index = self.kmeans.labels_ == self.clusterLabel[i]
      Gabriel_pairs = np.array(self.cbp.points)[midpoints_subset_index]
      subset_index = Gabriel_pairs.reshape(-1)
      model = svm.SVC(kernel='linear')
      model.fit(A_train[subset_index], self.C_train[subset_index])
      self.SVM.append(model)

  def ensemble(self, x):
    I = Euclidean_distance_vector(x,self.clusterCentroids)
    index = I.argsort()[0:self.ensembleNum]
    classifier = 0.0
    for i in range(len(index)):
      j = index[i]

      weight = 1/I[index[i]]/(np.sum(1/I[index]))
      classifier = classifier + weight*self.SVM[j].predict([x])
    if classifier >= 0:
      return 1
    else:
      return -1
  def predict(self,x):
    ensembleVec = np.vectorize(self.ensemble, signature = '(n)->()')
    predict = np.array(ensembleVec(np.array(x))).reshape(-1)
    #[1,-1] -> [0, 1]
    return self.transformer.transform_back(predict)



class LSVM(object):
  def __init__(self,A_train, C_train, args):
    self.A_train = A_train
    self.C_train = C_train
    self.K = args[0]
    self.C = args[1]

  def classifier(self,x):
    I = Euclidean_distance_vector(x,self.A_train)
    index = I.argsort()[0:self.K]
    label = np.unique(self.C_train[index])
    if len(label) == 1:
      return label[0]
    else:
      model = svm.SVC(kernel='linear', C = self.C)
      model.fit(self.A_train[index], self.C_train[index])
      return model.predict([x])

  def predict(self,x):
    classifiersVec = np.vectorize(self.classifier,  signature = '(n)->()')
    return classifiersVec(x)


class GLSVM(object):
  def __init__(self,A_train, C_train, args):
    self.cbp = []
    self.ensembleNum = args[1]
    self.K = args[0]
    self.SVM = []
    self.dist_matrix = []
    self.transformer = LabelEncode([-1,1])
    self.transformer.fit(C_train)
    self.C_train = self.transformer.transform(C_train)

    self._train(A_train, self.C_train)

  def _Euc(self, A_train):
    self.dist_matrix = np.zeros(shape=(len(A_train),len(A_train)))
    self.dist_matrix = np.array([Euclidean_distance_vector(A_i, A_train) for A_i in A_train])

  def _train(self, A_train, C_train):
    self.cbp = CBP(A_train, self.C_train)
    if self.ensembleNum > self.cbp.count:
      self.ensembleNum = self.cbp.count
    if self.K > self.cbp.count:
      self.K = self.cbp.count
    self._Euc(np.array(self.cbp.midpoints))
    for i in range(self.cbp.count):
      subsetIndex = self.dist_matrix[:,i].argsort()[0:self.K]
      Gabriel_pairs = np.array(self.cbp.points)[subsetIndex]
      subset_index = Gabriel_pairs.reshape(-1)
      model = svm.SVC(kernel='linear')
      model.fit(A_train[subset_index], self.C_train[subset_index])
      self.SVM.append(model)

  def ensemble(self, x):
    I = Euclidean_distance_vector(x,self.cbp.midpoints)
    index = I.argsort()[0:self.ensembleNum]
    classifier = 0.0
    for i in range(len(index)):
      j = index[i]

      weight = 1/I[index[i]]/(np.sum(1/I[index]))
      classifier = classifier + weight*self.SVM[j].predict([x])
    if classifier >= 0:
      return 1
    else:
      return -1
  def predict(self,x):
    ensembleVec = np.vectorize(self.ensemble, signature = '(n)->()')(x)
    predict = ensembleVec(x)
    #[1,-1] -> [0, 1]
    return self.transformer.transform_back(predict)










if __name__ == '__main__':
  main()
from model_utility import Generate_data_Curve, Euclidean_distance, Generate_data_Circle, Generate_data_special, Generate_data_tanh, Generate_data_Curve_noise, Generate_data_tanh2
from CBP import CBPClassifier, CBP, refrenced_method, GPSVM, GLSVM,LSVM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import random
from sklearn.model_selection import train_test_split, GridSearchCV
import contextlib
import io,time
import seaborn as sns
from sklearn import preprocessing
from brusselator import Brusselator_Data, dQ_dlambda, Brusselator_Data_Noise
from pathlib import Path
#models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from xgb_Caret import XGBoostClassifier 
from sklearn.svm import SVC


from scipy.io import arff


BootStrapRepeat = 5

def Gradient_Q1(x, y):
    return np.array([(1 - ( np.tanh(5*(y-(x-0.75)*(x-2)*x-0.75)) )**2)*5*(-(3*x**2 - 5.5*x + 1.5)), 
                     (1 - (np.tanh(5*(y-(x-0.75)*(x-2)*x-0.75)))**2)*5 ])
def Gradient_Q2(x, y):
    dx = 2*((y-.5*(np.tanh(20*x)*np.tanh(20*(x-.5))+1)*np.exp(.2*x**2)))*( 
        (y+1)*np.exp(0.2*x**2)*(0.4*x) +
        -0.5*(
                (0.4*x)*np.exp(0.2*x**2)*np.tanh(20*x)*np.tanh(20*(x-.5)) +
                np.exp(0.2*x**2)*( 1 - np.tanh(20*x)**2 )*20*np.tanh(20*(x-.5))+
                np.exp(0.2*x**2)*np.tanh(20*x)*(1 - np.tanh(20*(x-.5)))**2*(20)
        )
    )
    dy = 2*((y-.5*(np.tanh(20*x)*np.tanh(20*(x-.5))+1)*np.exp(.2*x**2)))*(np.exp(.2*x**2))
    return np.array([dx,dy])

####Single classifiers:
def G_PSVM_Binary(X_train, Y_train):
    n = len(Y_train)
    #kernel coeficients
    K = [ 1,3,5,7, 9, 12, 20]
    k = [1,3, 5]
    C = [0.1, 0.5,1, 10]
    parameters = [[i, j, c] for i in K for j in k for c in C]
    accMatrix = np.zeros((len(parameters),BootStrapRepeat))
    for j in range(BootStrapRepeat):
        trainIndex = np.unique(np.random.choice([k for k in range(n)],n))
        while len(np.unique(Y_train[trainIndex])) != len(np.unique(Y_train)):
            trainIndex = np.random.choice([k for k in range(n)],n)
        trainIndex  = np.unique(trainIndex)
        testIndex = [k for k in range(n) if k not in trainIndex]
        xTrain, yTrain = X_train[trainIndex], Y_train[trainIndex] 
        xTest, yTest = X_train[testIndex], Y_train[testIndex]
        model = GPSVM(xTrain, yTrain)
        for i in range(len(parameters)):
            model.fit(parameters[i])
            predicts = model.predict(xTest)
            accMatrix[i,j] = np.sum(predicts == yTest)/len(yTest)
    # optimal K
    optimalPara = parameters[accMatrix.mean(axis=1).argsort()[-1]]
    #predict
    print(optimalPara)
    model = GPSVM(X_train, Y_train)
    model.fit(optimalPara)
    return  model



def ECBP_PPR_base(X_train, Y_train):
    n = len(Y_train)
    #kernel coeficients
    r1 = [5, 10]
    degree1 = [3, 5]
    parameters = [[i, j, i, j] for i in r1 for j in degree1]
    accMatrix = np.zeros((len(parameters),BootStrapRepeat))
    for j in range(BootStrapRepeat):
        trainIndex = np.random.choice([k for k in range(n)],n)
        while len(np.unique(Y_train[trainIndex])) != len(np.unique(Y_train)):
            trainIndex = np.random.choice([k for k in range(n)],n)
        print(j)
        testIndex = [k for k in range(n) if k not in trainIndex]
        xTrain, yTrain = X_train[trainIndex], Y_train[trainIndex] 
        xTest, yTest = X_train[testIndex], Y_train[testIndex]
        model = CBPClassifier(xTrain, yTrain)
        for i in range(len(parameters)):
            #with contextlib.redirect_stdout(io.StringIO()):  
            model.fit(average = True, kind = 'PPR', args = parameters[i])
            predicts = model.predict(xTest)
            accMatrix[i,j] = np.sum(predicts == yTest)/len(yTest)
    # optimal K
    optimalPara = parameters[accMatrix.mean(axis=1).argsort()[-1]]
    #predict
    model = CBPClassifier(X_train, Y_train)
    with contextlib.redirect_stdout(io.StringIO()):  
        model.fit(average = False, kind = 'PPR', args = optimalPara)
    return  model

def Refrenced_method_base(X_train, Y_train):
    n = len(Y_train)
    #kernel coeficients
    alpha = [ 0,0.3,0.6]
    constLambda = [0.1,0.5,1,1.5]
    parameters = [[i, j] for i in alpha for j in constLambda]
    accMatrix = np.zeros((len(parameters),BootStrapRepeat))
    for j in range(BootStrapRepeat):
        trainIndex = np.random.choice([k for k in range(n)],n)
        while len(np.unique(Y_train[trainIndex])) != len(np.unique(Y_train)):
            trainIndex = np.random.choice([k for k in range(n)],n)
        testIndex = [k for k in range(n) if k not in trainIndex]
        xTrain, yTrain = X_train[trainIndex], Y_train[trainIndex] 
        xTest, yTest = X_train[testIndex], Y_train[testIndex]
        model = refrenced_method(xTrain, yTrain)
        print(j,'--',)
        for i in range(len(parameters)):
            print('--',i)
            model.fit(parameters[i][0], parameters[i][1])
            predicts = model.predict(xTest)
            accMatrix[i,j] = np.sum(predicts == yTest)/len(yTest)
    # optimal K
    optimalPara = parameters[accMatrix.mean(axis=1).argsort()[-1]]
    print(optimalPara)
    #predict
    model = refrenced_method(X_train, Y_train)
    model.fit(optimalPara[0], optimalPara[1])
    return  model

def L_SVM_base(X_train, Y_train):
    n = len(Y_train)
    #kernel coeficients
    K = [ 5,10,15,20]
    C = [0.1, 1, 10]
    parameters = [[i, j] for i in K for j in C]
    accMatrix = np.zeros((len(parameters),BootStrapRepeat))
    for i in range(len(parameters)):
        for j in range(BootStrapRepeat):
            trainIndex = np.random.choice([k for k in range(n)],n)
            while len(np.unique(Y_train[trainIndex])) != len(np.unique(Y_train)):
                trainIndex = np.random.choice([k for k in range(n)],n)
            testIndex = [k for k in range(n) if k not in trainIndex]
            xTrain, yTrain = X_train[trainIndex], Y_train[trainIndex] 
            xTest, yTest = X_train[testIndex], Y_train[testIndex]
            model = LSVM(xTrain, yTrain, parameters[i])
            predicts = model.predict(xTest)
            accMatrix[i,j] = np.sum(predicts == yTest)/len(yTest)
    # optimal K
    optimalPara = parameters[accMatrix.mean(axis=1).argsort()[-1]]
    print(optimalPara)
    #predict
    model = LSVM(X_train, Y_train, optimalPara)
    return  model

def ECBP_Gradient_base(X_train, Y_train):

    n = len(Y_train)
    #kernel coeficients
    K = [ 3,6,10]
    parameters = [[i] for i in K]
    gradientX = np.zeros((len(Y_train), len(X_train[0])))
    for i in range(len(Y_train)):
        gradientX[i]= dQ_dlambda(X_train[i][0], X_train[i][1])
        #gradientX[i]= Gradient_Q1(X_train[i][0], X_train[i][1])
        #gradientX[i]= Gradient_Q2(X_train[i][0], X_train[i][1])
    accMatrix = np.zeros((len(parameters),BootStrapRepeat))
    for i in range(len(parameters)):
        for j in range(BootStrapRepeat):
            trainIndex = np.random.choice([k for k in range(n)],n)
            while len(np.unique(Y_train[trainIndex])) != len(np.unique(Y_train)):
                trainIndex = np.random.choice([k for k in range(n)],n)
            testIndex = [k for k in range(n) if k not in trainIndex]
            xTrain, yTrain = X_train[trainIndex], Y_train[trainIndex] 
            xTest, yTest = X_train[testIndex], Y_train[testIndex]
            model = CBPClassifier(xTrain, yTrain)
            gradient = gradientX[trainIndex]
            model.fit(kind = 'gradient', args = [parameters[i], gradient])
            predicts = model.predict(xTest)
            accMatrix[i,j] = np.sum(predicts == yTest)/len(yTest)
    # optimal K
    optimalPara = parameters[accMatrix.mean(axis=1).argsort()[-1]]
    print(optimalPara)
    #predict
    model = CBPClassifier(X_train, Y_train)
    model.fit(kind = 'gradient', args = [optimalPara, gradientX])
    return model

def ECBP_binary(X_train, Y_train):
    model = CBPClassifier(X_train, Y_train)
    model.fit(kind = '1dSpline')
    return  model

####DAG
class multiclassClassifier(object):
    def __init__(self, model, X_train, Y_train):
        self.numLabel = len(np.unique(Y_train))
        self.labels = np.unique(Y_train)
        self.numNode = self.numLabel*(self.numLabel - 1) / 2
        self.left = None
        self.right = None
        self.model = None
        self._train(model, X_train, Y_train)

    def _train(self,model, X_train, Y_train):
        # make labels[0] vs labels[1]
        index = np.logical_or(np.array(Y_train) == self.labels[0],  np.array(Y_train) == self.labels[1] )
        xTrain = X_train[index]
        yTrain = Y_train[index]
        print('comparison')
        self.model = model(xTrain, yTrain)
        if self.numLabel > 2:
             # splt if label > 2
            # not label[1]
            index_1 = np.array(Y_train) != self.labels[1]
            self.left = multiclassClassifier(model, X_train[index_1], Y_train[index_1])
            # not label[0]
            index_0 = np.array(Y_train) != self.labels[0]
            self.right = multiclassClassifier(model, X_train[index_0], Y_train[index_0])

    def predictSingle(self, x):
        pred = self.model.predict([x])
        if(self.numLabel == 2):
            return(pred)
        else:
            if pred != self.labels[1]:
                return(self.left.predictSingle([x]))
            else:  
                return(self.right.predictSingle([x]))

    def predict(self, x):
        ensembleVec = np.vectorize(self.predictSingle, signature = '(n)->()')
        predict = ensembleVec(x)
        return predict
########################################################################################################################
########################################################################################################################
########################################################################################################################

def KNN(X_train, Y_train, X_predict, Y_predict):
    #tunning
    K = [1,3,5,7]
    n = len(Y_train)
    accMatrix = np.zeros((len(K),BootStrapRepeat))
    for i in range(len(K)):
        for j in range(BootStrapRepeat):
            trainIndex = np.random.choice([k for k in range(n)],n)
            while len(np.unique(Y_train[trainIndex])) != len(np.unique(Y_train)):
                trainIndex = np.random.choice([k for k in range(n)],n)
            testIndex = [k for k in range(n) if k not in trainIndex]
            xTrain, yTrain = X_train[trainIndex], Y_train[trainIndex] 
            xTest, yTest = X_train[testIndex], Y_train[testIndex]

            KNN_model = KNeighborsClassifier(n_neighbors = K[i]).fit(xTrain,yTrain)
            predicts = KNN_model.predict(xTest)
            accMatrix[i,j] = np.sum(predicts == yTest)/len(yTest)
    # optimal K
    optimalPara = K[accMatrix.mean(axis=1).argsort()[-1]]
    print(optimalPara)
    #predict
    KNN_model = KNeighborsClassifier(n_neighbors = optimalPara).fit(X_train, Y_train)
    predicts = KNN_model.predict(X_predict)
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return accuracy

def rf(X_train, Y_train, X_predict, Y_predict):
    n = len(Y_train)
    max_depth = [2,4,8, 12]
    max_features = [1,2] #np.linspace(1, np.sqrt(len(X_train[0])), 5)
    parameters = [[i,j] for i in max_depth for j in max_features]
    accMatrix = np.zeros((len(parameters),BootStrapRepeat))
    for i in range(len(parameters)):
        for j in range(BootStrapRepeat):
            trainIndex = np.random.choice([k for k in range(n)],n)
            while len(np.unique(Y_train[trainIndex])) != len(np.unique(Y_train)):
                trainIndex = np.random.choice([k for k in range(n)],n)
            testIndex = [k for k in range(n) if k not in trainIndex]
            xTrain, yTrain = X_train[trainIndex], Y_train[trainIndex] 
            xTest, yTest = X_train[testIndex], Y_train[testIndex]
            clf_model = RandomForestClassifier(max_depth = parameters[i][0], max_features = parameters[i][1]).fit(xTrain,yTrain)
            predicts = clf_model.predict(xTest)
            accMatrix[i,j] = np.sum(predicts == yTest)/len(yTest)
    # optimal K
    optimalPara = parameters[accMatrix.mean(axis=1).argsort()[-1]]
    print(optimalPara)
    #predict
    clf_model = RandomForestClassifier(max_depth = optimalPara[0], max_features = optimalPara[1]).fit(X_train, Y_train)
    predicts = clf_model.predict(X_predict)
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return  accuracy

def nnet(X_train, Y_train, X_predict, Y_predict):
    n = len(Y_train)
    alpha = [0.001,0.01,0,1]
    learningRate = [0.001,0.01,0.1]
    parameters = [[i,j] for i in alpha for j in learningRate]
    accMatrix = np.zeros((len(parameters),BootStrapRepeat))
    for i in range(len(parameters)):
        for j in range(BootStrapRepeat):
            trainIndex = np.random.choice([k for k in range(n)],n)
            while len(np.unique(Y_train[trainIndex])) != len(np.unique(Y_train)):
                trainIndex = np.random.choice([k for k in range(n)],n)
            testIndex = [k for k in range(n) if k not in trainIndex]
            xTrain, yTrain = X_train[trainIndex], Y_train[trainIndex] 
            xTest, yTest = X_train[testIndex], Y_train[testIndex]
            model = MLPClassifier(alpha = parameters[i][0], learning_rate_init = parameters[i][1], max_iter = 5000).fit(xTrain,yTrain)
            predicts = model.predict(xTest)
            accMatrix[i,j] = np.sum(predicts == yTest)/len(yTest)
    # optimal K
    optimalPara = parameters[accMatrix.mean(axis=1).argsort()[-1]]
    print(optimalPara)
    #predict
    model = MLPClassifier(alpha = optimalPara[0], learning_rate_init = optimalPara[1], max_iter = 5000).fit(X_train, Y_train)
    predicts = model.predict(X_predict)
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return  accuracy

def xgBoost(X_train, Y_train, X_predict, Y_predict):
    num_class = len(np.unique(Y_train) )
    clf = XGBoostClassifier(
        eval_metric = 'auc',
        num_class = num_class,
        nthread = 4,
        silent = 1,
        )
    parameters = {
        'num_boost_round': [100, 250, 500],
        'eta': [0.05, 0.1, 0.3],
        'max_depth': [6, 9, 12],
        'subsample': [0.9, 1.0],
        'colsample_bytree': [0.9, 1.0],
    }
    with contextlib.redirect_stdout(io.StringIO()): 
        clf = GridSearchCV(clf, parameters, n_jobs=1, cv=5)
        clf.fit(X_train, Y_train)
    best_parameters = clf.best_estimator_
    predicts = clf.predict(X_predict)
    return  np.sum(predicts == Y_predict)/len(Y_predict)

def SVM_RBF(X_train, Y_train, X_predict, Y_predict):
    n = len(Y_train)
    #kernel coeficients
    gamma = [ 0.1, 0.5, 0.7,0.8,0.9,1,1.2, 1.5]
    parameters = [[i] for i in gamma]
    accMatrix = np.zeros((len(gamma),BootStrapRepeat))
    for i in range(len(gamma)):
        for j in range(BootStrapRepeat):
            trainIndex = np.random.choice([k for k in range(n)],n)
            while len(np.unique(Y_train[trainIndex])) != len(np.unique(Y_train)):
                trainIndex = np.random.choice([k for k in range(n)],n)
            testIndex = [k for k in range(n) if k not in trainIndex]
            xTrain, yTrain = X_train[trainIndex], Y_train[trainIndex] 
            xTest, yTest = X_train[testIndex], Y_train[testIndex]
            model = SVC( gamma = parameters[i][0], kernel = 'rbf').fit(xTrain,yTrain)
            predicts = model.predict(xTest)
            accMatrix[i,j] = np.sum(predicts == yTest)/len(yTest)
    # optimal K
    optimalPara = parameters[accMatrix.mean(axis=1).argsort()[-1]]
    print(optimalPara)
    #predict
    model = SVC(gamma = optimalPara[0], kernel = 'rbf').fit(xTrain,yTrain)
    predicts = model.predict(X_predict)
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return  accuracy

def SVM_kernel(X_train, Y_train, X_predict, Y_predict):
    param_grid = {'C': [0.1, 1, 10],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'poly']}
 
    grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
    # fitting the model for grid search
    grid.fit(X_train, Y_train)
    predicts = grid.predict(X_predict)
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return  accuracy
def SVM_Linear(X_train, Y_train, X_predict, Y_predict):
    model = SVC(kernel = 'linear').fit(X_train, Y_train)
    predicts = model.predict(X_predict)
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return  accuracy
#####------------------ Binary classifiers ------------------------###
#####------------------ Binary classifiers ------------------------###
#####------------------ Binary classifiers ------------------------###





def ECBP(X_train, Y_train, X_predict, Y_predict):
    model = multiclassClassifier(ECBP_binary, X_train, Y_train)
    predicts = model.predict(X_predict)
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return  accuracy

def ECBP_avg(X_train, Y_train, X_predict, Y_predict):
    model = CBPClassifier(X_train, Y_train)
    model.fit(average = True, kind = '1dSpline')
    predicts = model.predict(X_predict)
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return  accuracy

def ECBP_PPR(X_train, Y_train, X_predict, Y_predict):
    model = multiclassClassifier(ECBP_PPR_base, X_train, Y_train)
    predicts = model.predict(X_predict)
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return  accuracy

def Refrenced_method(X_train, Y_train, X_predict, Y_predict):
    model = multiclassClassifier(Refrenced_method_base, X_train, Y_train)
    predicts = model.predict(X_predict)
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return  accuracy

def L_SVM(X_train, Y_train, X_predict, Y_predict):
    model = multiclassClassifier(L_SVM_base, X_train, Y_train)
    predicts = model.predict(X_predict)
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return  accuracy

def G_PSVM(X_train, Y_train, X_predict, Y_predict):
    model = multiclassClassifier(G_PSVM_Binary, X_train, Y_train)
    predicts = model.predict(X_predict)
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return  accuracy


def ECBP_Gradient(X_train, Y_train, X_predict, Y_predict):
    model = multiclassClassifier(ECBP_Gradient_base, X_train, Y_train)
    predicts = model.predict(X_predict)
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return  accuracy
########################################################################################################################
########################################################################################################################
########################################################################################################################




def time_Comp(N, k, repeat = 10):
    time1 = np.zeros((len(N), k))
    time2 = np.zeros((len(N), k))
    for i in range(len(N)):
        print(i)
        for j in range(repeat):
            A = Generate_data_Curve(N[i])
            A_train = A.iloc[:,0:2].values
            C_train = A.iloc[:,2].values
            X_train,  X_predict, Y_train, Y_predict = train_test_split(A_train, C_train, test_size=0.8, random_state=42)
            start =  time.time()
            p = G_LSVM(X_train, Y_train, X_predict, Y_predict)
            stop =  time.time()
            time1[i,j] = stop - start

            start =  time.time()
            p = L_SVM(X_train, Y_train, X_predict, Y_predict)
            stop =  time.time()
            time2[i,j] = stop - start
    df = pd.DataFrame( {'Kmeans':time1[:,:repeat].reshape(-1), 'CBP+Kmeans':time2[:,:repeat].reshape(-1),
                    'n':np.repeat(N, repeat)} )
    df_plot = df.melt(id_vars='n', value_vars=['Kmeans', "CBP+Kmeans"])
    graph = sns.boxplot(x='n', y='value', hue='variable', data=df_plot)
    plt.show()
    return

def accuracy_incre_compare(N, outname, testsize = 1000, repeat = 20):
    # KNN,rf, nnet, xgboost, SVM_RBF,SVM_poly,SVM_Linear,  ECBP, ECBP_avg, ECBP_PPR_avg, L_SVM, G_PSVM, G_LSVM, Referenced method
    # N train size
    names = ['KNN','rf', 'nnet', 'xgboost', 'SVM_kernel','SVM_Linear',
       'ECBP','ECBP_PPR', 'L_SVM', 'G_PSVM', 'Referenced method', 'Gradient_CBP']
    classifier = [KNN, rf, nnet, xgBoost,SVM_kernel,SVM_Linear,ECBP,ECBP_PPR,L_SVM,G_PSVM,Refrenced_method, ECBP_Gradient]
    accu_matrix = np.zeros((len(N), len(names)))
    for i in range(len(N)):
        for r in range(repeat):
            print(i,'----',r)
            A = Brusselator_Data(testsize + N[i], 2)
            xTrain = A.iloc[0:N[i],0:2].values
            yTrain = A.iloc[0:N[i],2].values
            xTest = A.iloc[N[i]:,0:2].values
            yTest = A.iloc[N[i]:,2].values
            for j in range(len(classifier)):
                accu_matrix[i][j] = accu_matrix[i][j] + classifier[j](xTrain, yTrain, xTest,yTest)/repeat
    np.savetxt(outname,accu_matrix, delimiter=",", header = '')
    return

def accuracy_incre_compare_func(function, N, outname, testsize = 1000, repeat = 20):
    # KNN,rf, nnet, xgboost, SVM_RBF,SVM_poly,SVM_Linear,  ECBP, ECBP_avg, ECBP_PPR_avg, L_SVM, G_PSVM, G_LSVM, Referenced method
    # N train size
    names = ['KNN','rf', 'nnet', 'xgboost', 'SVM_kernel','SVM_Linear',
       'ECBP_PPR', 'L_SVM', 'G_PSVM', 'Referenced method', 'Gradient_CBP']
    classifier = [KNN, rf, nnet, xgBoost,SVM_kernel,SVM_Linear,ECBP_PPR,L_SVM,G_PSVM,Refrenced_method, ECBP_Gradient]
    accu_matrix = np.zeros((len(N), len(names)))
    for i in range(len(N)):
        for r in range(repeat):
            print(i,'----',r)
            A = function(testsize + N[i])
            xTrain = A.iloc[0:N[i],0:2].values
            yTrain = A.iloc[0:N[i],2].values
            xTest = A.iloc[N[i]:,0:2].values
            yTest = A.iloc[N[i]:,2].values
            for j in range(len(classifier)):
                accu_matrix[i][j] = accu_matrix[i][j] + classifier[j](xTrain, yTrain, xTest,yTest)/repeat
    np.savetxt(outname,accu_matrix, delimiter=",", header = '')
    return


def relative(col):
    return col/max(col)

def relative_min(col):
    return col/min(col)

def accuracy_comp(data,  savefilename, repeat = 5, folds = 5):
    # KNN,rf, nnet, xgboost, SVM_RBF,SVM_poly,SVM_Linear,  ECBP, ECBP_avg, ECBP_PPR_avg, L_SVM, G_PSVM, G_LSVM, Referenced method
    # 
    A = data
    N = len(A)
    names = ['KNN','rf', 'nnet',  'xgBoost', 'SVM_kernel','SVM_Linear',
       'ECBP_PPR', 'L_SVM', 'G_PSVM', 'Referenced method']
    indexes = []
    accuracy = np.zeros((repeat*folds, len(names)))
    classifier = [KNN, rf, nnet, xgBoost, SVM_kernel, SVM_Linear,ECBP_PPR,L_SVM,G_PSVM,Refrenced_method]
    index = [int(N/folds)*i for i in range(folds)]
    index.append(N)
    numClass = len(np.unique(A.iloc[:,-1]))
    for i in range(5):
        indexes.append(np.arange(index[i], index[i+1]))
    for r in range(repeat):
        A = A.sample(frac= 1, replace = False).reset_index(drop=True)
        A_train = A.iloc[:,0:-1].values
        C_train = A.iloc[:,-1].values
        #
        for i in range(folds):
            print(r,'----',i)
            IndexTest= indexes[i]
            IndexTrain  = [k for k in range(N) if k not in IndexTest]
            xTrain = A_train[IndexTrain]
            yTrain = C_train[IndexTrain]
            xTest = A_train[IndexTest]
            yTest = C_train[IndexTest]
            for j in range(len(classifier)):
                print('initiating',names[j])
                accuracy[r*folds + i][j] = classifier[j](xTrain, yTrain, xTest,yTest)
                print('ended',names[j])
    np.savetxt(savefilename, accuracy, delimiter=",", header = '')
    return

def accuracy_comp_brusselator(N , testsize , savefilename, repeat = 20, label = 2):
    # KNN,rf, nnet, xgboost, SVM_RBF,SVM_poly,SVM_Linear,  ECBP, ECBP_avg, ECBP_PPR_avg, L_SVM, G_PSVM, G_LSVM, Referenced method
    # 
    names = ['KNN','rf', 'nnet',  'xgBoost', 'SVM_kernel','SVM_Linear',
       'setBoundary_1dInte','setBoundary_PPR', 'L_SVM', 'G_PSVM', 'Referenced method', 'Gradient_CBP']
    accuracy = np.zeros((repeat, len(names)))
    classifier = [KNN, rf, nnet, xgBoost, SVM_kernel, SVM_Linear, ECBP, ECBP_PPR,L_SVM,G_PSVM,Refrenced_method, ECBP_Gradient]
    for r in range(repeat):
        print(r,'----')
        A = Brusselator_Data(testsize + N, sep = label)
        xTrain = A.iloc[0:N,0:-1].values
        yTrain = A.iloc[0:N,-1].values
        xTest = A.iloc[N:,0:-1].values
        yTest = A.iloc[N:,-1].values
        for j in range(len(classifier)):
            print('initiating',names[j])
            accuracy[r][j] = classifier[j](xTrain, yTrain, xTest,yTest)
            print('ended',names[j])
    np.savetxt(savefilename, accuracy, delimiter=",", header = '')
    return

def accuracy_comp_function(function, N , testsize , savefilename, repeat = 20):
    # KNN,rf, nnet, xgboost, SVM_RBF,SVM_poly,SVM_Linear,  ECBP, ECBP_avg, ECBP_PPR_avg, L_SVM, G_PSVM, G_LSVM, Referenced method
    # 
    names = ['KNN','rf', 'nnet',  'xgBoost', 'SVM_kernel','SVM_Linear',
       'ECBP_PPR', 'L_SVM', 'G_PSVM', 'Referenced method', 'Gradient_CBP']
    accuracy = np.zeros((repeat, len(names)))
    classifier = [KNN, rf, nnet, xgBoost, SVM_kernel, SVM_Linear,ECBP_PPR,L_SVM,G_PSVM,Refrenced_method, ECBP_Gradient]
    for r in range(repeat):
        print(r,'----')
        A = function(testsize + N)
        xTrain = A.iloc[0:N,0:-1].values
        yTrain = A.iloc[0:N,-1].values
        xTest = A.iloc[N:,0:-1].values
        yTest = A.iloc[N:,-1].values
        for j in range(len(classifier)):
            print('initiating',names[j])
            accuracy[r][j] = classifier[j](xTrain, yTrain, xTest,yTest)
            print('ended',names[j])
    np.savetxt(savefilename, accuracy, delimiter=",", header = '')
    return

 ########################################  ########################################
 ######################################## ########################################
 ################################################################################
  #########################################plot functions

def plot_accuracy(filename, outfile):
    names = ['KNN','random forest', 'nnet',  'xgBoost', 'SVM_kernel','SVM_Linear'
     ,'setBounday_PPR', 'L_SVM', 'G_PSVM', 'Referenced method', 'Gradient']
    df = pd.read_csv(filename, header = None, names = names)
    plt.subplot(1,2,1)
    plt.ylim([0,1])
    df.boxplot(fontsize = 5,rot=30)
    plt.ylabel('Accuracy')
    plt.xticks(fontsize = 10)
    plt.subplot(1,2,2)
    plt.ylim([0,1])
    df = df.apply(relative, axis = 1)
    df.boxplot(fontsize = 5,rot=30)
    plt.ylabel('Relative Accuracy')
    plt.axhline(1, c = 'red')
    plt.gcf().set_size_inches(20,10)
    plt.xticks(fontsize = 10)
    plt.savefig(Path(outfile), bbox_inches='tight')
    #plt.show()

def plot_accuracy_inv(filename, outfile):
    names = ['KNN','random forest', 'nnet',  'xgBoost', 'SVM_kernel','SVM_Linear'
    , 'setBounday_PPR', 'L_SVM', 'G_PSVM', 'Referenced method', 'Gradient']
    df = pd.read_csv(filename, header = None, names = names)
    print(df)
    df = df.apply(lambda x: 1-x, axis = 0)
    print(df)
    plt.subplot(1,2,1)
    plt.ylim([0,1])
    df.boxplot(fontsize = 5,rot=30)
    plt.ylabel('Misclassification')
    plt.xticks(fontsize = 10)
    plt.subplot(1,2,2)
    df = df.apply(relative_min, axis = 1)
    print(df)
    df.boxplot(fontsize = 5,rot=30)
    plt.ylabel('Relative Misclassification')
    plt.axhline(1, c = 'red')
    plt.gcf().set_size_inches(20,10)
    plt.xticks(fontsize = 10)
    plt.savefig(Path(outfile), bbox_inches='tight')

def plot_accuracy_comp_inc(filename, N):
    names = ['KNN','random forest', 'nnet',  'xgBoost','SVM_kernel','SVM_Linear',
     'setBounday_PPR', 'L_SVM', 'G_PSVM', 'Referenced method', 'Gradient']
    df = pd.read_csv(filename, header = None, names = names)
    df['n'] = N
    df.plot(x = 'n',kind = 'line', style='.-', cmap = 'turbo')
    plt.ylim([0,1])
    plt.ylabel('Accuracy')
    plt.xlabel('Training size')
    plt.gcf().set_size_inches(10,10)
    plt.savefig(Path('results/accu_vs_sample_brusslator.png'), bbox_inches='tight')
    return

def plot_accuracy_comp_inc_partial(filename, N):
    names = ['KNN','random forest', 'nnet',  'xgBoost','SVM_kernel','SVM_Linear',
     'setBounday_PPR', 'L_SVM', 'G_PSVM', 'Referenced method', 'Gradient']
    df = pd.read_csv(filename, header = None, names = names)
    df['n'] = N
    df = df.iloc[0:8,:]
    fig, ax = plt.subplots()
    df.plot(x = 'n',y = names ,kind = 'line', style='-.', cmap = 'turbo', ax = ax)
    for line in ax.get_lines():
        if line.get_label() in ['G_PSVM','setBounday_PPR', 'Gradient']:
            line.set_linewidth(2)
            line.set_linestyle('-')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Training size')
    plt.gcf().set_size_inches(10,10)
    plt.savefig(Path('results/accu_vs_sample_brusslator_partial.png'), bbox_inches='tight')
    return

def plot_accuracy_comp_inc_inv(filename, N):
    names = ['KNN','random forest', 'nnet',  'xgBoost','SVM_kernel','SVM_Linear',
     'setBounday_PPR', 'L_SVM', 'G_PSVM', 'Referenced method', 'Gradient']
    df = pd.read_csv(filename, header = None, names = names)
    df['n'] = N
    df.iloc[:,:-1] = df.iloc[:,:-1].apply(lambda x: 1-x, axis = 0)
    df.plot(x = 'n',kind = 'line', style='.-', cmap = 'turbo')
    plt.ylim([0,1])
    plt.ylabel('Misclassification rate')
    plt.xlabel('Training size')
    plt.gcf().set_size_inches(10,10)
    plt.savefig(Path('results/accu_vs_sample_brusslator_inv.png'), bbox_inches='tight')
    return

def plot_accuracy_comp_inc_partial_inv(filename, N):
    names = ['KNN','random forest', 'nnet',  'xgBoost','SVM_kernel','SVM_Linear',
     'setBounday_PPR', 'L_SVM', 'G_PSVM', 'Referenced method', 'Gradient']
    df = pd.read_csv(filename, header = None, names = names)
    df['n'] = N
    df = df.iloc[0:8,:]
    df.iloc[:,:-1] = df.iloc[:,:-1].apply(lambda x: 1-x, axis = 0)
    fig, ax = plt.subplots()
    df.plot(x = 'n',y = names ,kind = 'line', style='-.', cmap = 'turbo', ax = ax)
    for line in ax.get_lines():
        if line.get_label() in ['G_PSVM','setBounday_PPR', 'Gradient']:
            line.set_linewidth(2)
            line.set_linestyle('-')
    plt.legend()
    plt.ylabel('Misclassification rate')
    plt.xlabel('Training size')
    plt.gcf().set_size_inches(10,10)
    plt.savefig(Path('results/accu_vs_sample_brusslator_partial_inv.png'), bbox_inches='tight')
    return



################################################### data ############################################
 ######################################## ########################################
def abalone_Data():
    abalone = pd.read_csv('data/abalone.data', header = None)
    abalone = abalone[abalone.iloc[:,0] != 'I']
    A_train = abalone.iloc[:,:-1]  # preprocessing.normalize(abalone.iloc[:,1:].values)
    C_train = abalone.iloc[:,-1]
    A_train['F'] = [int(i) for i in A_train.iloc[:,0] == 'F']
    A_train['M'] = [int(i) for i in A_train.iloc[:,0] == 'M']
    A_train['I'] = [int(i) for i in A_train.iloc[:,0] == 'I']    
    A_train = A_train.iloc[:,1:].values
    C_train = [int(i) for i in C_train > 10] 
    data = pd.DataFrame(A_train)
    data['Z'] = C_train
    return data

def Iris_Data():
    #ckeck 1
    df = pd.read_csv('data/iris.data', header = None)
    df.iloc[:,:-1] = preprocessing.normalize(df.iloc[:,:-1].values)
    C_train = df.iloc[:,-1]   
    le = preprocessing.LabelEncoder()
    le.fit(df.iloc[:,-1] )
    df.iloc[:,-1]  = le.transform(df.iloc[:,-1] )
    return df

def breastCancer_Data():
    # check 2
    df = pd.read_csv('data/breast-cancer-wisconsin.data', header = None)
    df.iloc[:,6] = df.iloc[:,6].replace('?', 'NaN').astype(float)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # drop rows with NaNs
    df = df.dropna()
    df.iloc[:,6] =  pd.to_numeric(df.iloc[:,6])
    A_train = df.iloc[:,1:-1]
    A_train = preprocessing.normalize(A_train.values)
    C_train = df.iloc[:,-1]   
    data = pd.DataFrame(A_train)
    data['Z'] = C_train
    return data

def Diabetic_Data():
    #too long/ no
    data = arff.loadarff('data/messidor_features.arff')
    df = pd.DataFrame(data[0])
    A_train = df.iloc[:,0:-1] 
    #A_train = preprocessing.normalize(A_train)
    le = preprocessing.LabelEncoder()
    le.fit(df.iloc[:,-1] )
    Z  = le.transform(df.iloc[:,-1] )
    df = pd.DataFrame(A_train)
    df['Z'] = Z
    return df

def Bupa_liver_Data():
    #check 3
    df = pd.read_csv('data/bupa.data', header = None)
    A_train = df.iloc[:,0:-1] 
    A_train = preprocessing.normalize(A_train)
    le = preprocessing.LabelEncoder()
    le.fit(df.iloc[:,-1] )
    Z  = le.transform(df.iloc[:,-1] )
    df = pd.DataFrame(A_train)
    df['Z'] = Z
    return df

def Heart_Disease_Dat():
    # no
    df = pd.read_csv('data/processed.cleveland.data', header = None)
    df = df.replace('?', 'NaN').astype(float)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # drop rows with NaNs
    df = df.dropna()
    A_train = df.iloc[:,0:-1] 
    A_train = preprocessing.normalize(A_train)
    le = preprocessing.LabelEncoder()
    le.fit(df.iloc[:,-1] )
    Z  = le.transform(df.iloc[:,-1] )
    df = pd.DataFrame(A_train)
    df['Z'] = Z
    return df

def ForestFire_Data():
    #check 4
    df = pd.read_csv('data/Algerian_forest_fires_dataset_UPDATE.csv', header = None, skiprows=[0,1,124,125,126,170])
    A_train = df.iloc[:,3:-1].apply(pd.to_numeric)
    A_train = preprocessing.normalize(A_train)
    le = preprocessing.LabelEncoder()
    le.fit(df.iloc[:,-1] )
    Z  = le.transform(df.iloc[:,-1] )
    df = pd.DataFrame(A_train)
    df['Z'] = Z < 3
    return df

def NewThyroid_Data():
    #check 5
    data = pd.read_csv('data/new-thyroid.data', header = None)
    A_train = data.iloc[:,1:]
    A_train = preprocessing.normalize(A_train)
    df = pd.DataFrame(A_train)
    df['Z'] = data.iloc[:,0]
    return df

def Statlog_Heard_Data():
    #check 6
    data = pd.read_csv('data/heart.dat', sep=' ',header = None)
    data.iloc[:,:-1] = preprocessing.normalize(data.iloc[:,:-1])
    return data

def Statlog_Au_Data():
    #check 7
    data = pd.read_csv('data/australian.dat', sep=' ',header = None)
    data.iloc[:,:-1] = preprocessing.normalize(data.iloc[:,:-1])
    return data

def Inosphere_Data():
    #check 8
    data = pd.read_csv('data/ionosphere.data',header = None)
    data.iloc[:,:-1] = preprocessing.normalize(data.iloc[:,:-1])
    A_train = data.iloc[:,0:-1] 
    le = preprocessing.LabelEncoder()
    le.fit(data.iloc[:,-1] )
    Z  = le.transform(data.iloc[:,-1] )
    df = pd.DataFrame(A_train)
    df['Z'] = Z
    return df

    return data

def Amphibians_Data():
    #check 9
    data = pd.read_csv('data/amphibians.csv',header = None, skiprows=[0,1], sep = ';')
    A_train = data.iloc[:,2:]
    A_train = preprocessing.normalize(A_train)
    le = preprocessing.LabelEncoder()
    le.fit(data.iloc[:,-1] )
    Z  = le.transform(data.iloc[:,-1] )
    df = pd.DataFrame(A_train)
    df['Z'] = Z
    return df

def Bank_Data():
    data = pd.read_csv('data/data_banknote_authentication.txt',header = None, skiprows=[0,1], sep = ',')
    data.iloc[:,:-1] = preprocessing.normalize(data.iloc[:,:-1])
    return data

def Audit_Data():
    data = pd.read_csv('data/audit_risk.csv',header = None, skiprows=[0,1])
    data = data.drop(1, axis  = 1)
    print(data.dtypes)
    print(data)
    data = data.dropna()
    data.iloc[:,:-1] = preprocessing.normalize(data.iloc[:,:-1])
    return data
########################################################################

def main():
    N = [20,30,40,50,60,70,80,100, 150,200,300, 500, 800, 1200]
    #plot__box_plot(n = 100)


    #A = Iris_Data()
    #A = abalone_Data()
    #A = breastCancer_Data()
    #A = Bupa_liver_Data()
    #A = ForestFire_Data()
    #print(A)
    #A = NewThyroid_Data()
    #A = Statlog_Heard_Data()
    # A = Statlog_Au_Data()
    # print(A)
    #A = Inosphere_Data()
    #A = Amphibians_Data()
    #A = Bank_Data()
    #A = Audit_Data()
    #print(A)
    #accuracy_comp(A, savefilename = 'results/breastCancer_data.csv')
    #print(A)
    # print('Bupa -- might not work')
    # A = Bupa_liver_Data()
    # print(A)
    # accuracy_comp(A, savefilename = 'results/Bupa_liver_data.csv')

    # top left terminal
    # A = ForestFire_Data()
    # accuracy_comp(A, savefilename = 'results/ForestFire_data.csv')

    # top right terminal


    #top left terminal
    # A = abalone_Data()
    # accuracy_comp(A, savefilename = 'results/Abalone_data.csv')

    #top right terminal
    # A = NewThyroid_Data()
    # print(np.sum(A.iloc[:,-1] == 1), np.sum(A.iloc[:,-1] == 2), np.sum(A.iloc[:,-1] == 3))
    # print(A)
    # accuracy_comp(A, savefilename = 'results/NewThyroid_data.csv')

    #bottom left terminal
    # A = Statlog_Heard_Data()
    # accuracy_comp(A, savefilename = 'results/Statlog_Heard_data.csv')
    
    #top left terminal
    # A = Statlog_Au_Data()
    # accuracy_comp(A, savefilename = 'results/Statlog_Au.csv')


    # df = pd.read_csv('results/Breast_data.csv', header = None)
    # #df = pd.read_csv('results/Iris_data.csv', header = None)
    # #df = pd.read_csv('results/NewThyroid_data.csv', header = None)
    # #df = pd.read_csv('results/Statlog_Heard_data.csv', header = None)
    # #df = pd.read_csv('results/ForestFire_data.csv', header = None)
    # df = pd.read_csv('results/Bupa_liver_data.csv', header = None)
    # df = pd.read_csv('results/Inosphere_data.csv', header = None)
    # # df = pd.read_csv('results/Amphibians_data.csv', header = None)
    # df = pd.read_csv('results/Bank.csv', header = None)
    # #df = pd.read_csv('results/Amphibians_data.csv', header = None)
    # df =pd.read_csv('results/Statlog_Au.csv', header = None)
    # print(df.mean(axis = 0).round(3))
    # print(df.std(axis = 0).round(3))
    # data = df.apply(relative, axis = 1)
    # print(data.mean(axis = 0).round(3))
    # return
    #bot left
    # A = Inosphere_Data()
    # accuracy_comp(A, savefilename = 'results/Inosphere_data.csv')
    
    #A = Amphibians_Data()
    #accuracy_comp(A, savefilename = 'results/Amphibians_data.csv')

    #A = Bank_Data()
    #accuracy_comp(A, savefilename = 'results/Bank.csv')


    #A = Brusselator_Data(100, 2)
    #A = Generate_data_tanh(100)


    
    #accuracy_comp_brusselator(N = 100, testsize = 1000,savefilename = 'results/Brusselator_Accuracy_data_100_2_1d.csv', repeat = 20, label = 2)
    #plot_accuracy("results/Brusselator_Accuracy_data_100_2_1d.csv", 'results/brusselator_100_2_1d.png')
    
    #accuracy_comp(A, savefilename = 'results/Brusselator_Accuracy_data_200_2.csv')
    # plot_accuracy("results/Brusselator_Accuracy_data_100_2.csv")
    #plot_accuracy("results/Brusselator_Accuracy_data_200_2.csv")

    #accuracy_incre_compare(N, outname = 'results/Accuracy_vs_sample_brusselator.csv')
    # plot_accuracy_comp_inc("results/Accuracy_vs_sample_brusselator.csv", N)
    # plot_accuracy_comp_inc_partial("results/Accuracy_vs_sample_brusselator.csv", N)
    # plot_accuracy_comp_inc_inv("results/Accuracy_vs_sample_brusselator.csv", N)
    # plot_accuracy_comp_inc_partial_inv("results/Accuracy_vs_sample_brusselator.csv", N)
    A = Brusselator_Data_Noise(200)
    plt.scatter(A.iloc[:,0],A.iloc[:,1], c = A.iloc[:,2], cmap = 'turbo')
    plt.show()


    #plot_accuracy('results/Irise_Accuracy_data.csv')
    # print(A['Z'].values)
    # A = Generate_data_tanh(100)
    # A = Brusselator_Data_Noise(200,0.1)
    # model = multiclassClassifier(G_PSVM_Binary, A.iloc[:,0:-1].values, A.iloc[:,-1].values)
    # #model_PPR = multiclassClassifier(Refrenced_method_base, A_train, C_train)
    # B = Brusselator_Data_Noise(1000,0.1)
    # pred = model.predict(B.iloc[:,0:-1].values)
    # print(np.sum(pred == B.iloc[:,-1])/len(B.iloc[:,-1]))

    # plt.subplot(1,2,1)
    # # plt.scatter(X_train[:,0],X_train[:,1], c = Y_train, cmap = 'turbo')

    # pred = rf(X_train, Y_train, X_predict,Y_predict )
    # print(pred)
    # # print('-----')
    # pred = model.predict(X_predict)
    # p = np.sum(pred == Y_predict)/len(Y_predict)
    # print(p)
    # plt.subplot(1,2,2)
    # # plt.scatter(X_predict[:,0],X_predict[:,1], c = pred, cmap = 'turbo')
    # # #accuracy_comp_plot(400, repeat = 10, folds = 5)
    # # #plot_accuracy_comp_inc()
    # plt.show()
    #accuracy_comp_plot_data(data = Iris_Data(), filename = 'Iris_data_result.csv' ,repeat = 10, folds = 5)
    
    #time_Comp([200,300,400,500,600,700,1000],k = 10,repeat = 10)
    #accuracy_incre(N = [20,40,60,80,150,200,300, 500])
    #plot_accuracy("accuracy_results.csv")
    #abalone_Data()
    #plot_accuracy_inv("results/Brusselator_Accuracy_data_100_2.csv", "results/brusselator_100_2_inv.png")

    # plt.show()
    # A = Generate_data_Curve(200)
    # accuracy_comp_data(data = A,  filename = 'Curve_200.csv', repeat = 10, folds = 5)
    

    # plot_accuracy_comp_inc("results/accuracy_sample_brusselator_binary.csv", N)
    #accuracy_incre(N = (30,40,60,80,100,150,200,250,500))
    #N = [20,30,40,50,60,70,80,100, 150,200,300, 500, 800, 1200]
    
    # accuracy_comp_function(Generate_data_tanh, 100 , 1000 , 'results/tanh_100.csv', repeat = 10)
    # plot_accuracy("results/tanh_100", 'results/Plots/tanh_100.png')
    # plot_accuracy_inv("results/tanh_100", 'results/Plots/tanh_100_inv.png')

    # accuracy_comp_function(Generate_data_tanh, 100 , 1000 , 'results/tanh_100.csv', repeat = 10)
    #plot_accuracy("results/tanh_100.csv", 'results/Plots/tanh_100.png')
    #plot_accuracy_inv("results/tanh_100.csv", 'results/Plots/tanh_100_inv.png')

    #accuracy_comp_function(Generate_data_tanh2, 100 , 1000 , 'results/tanh_1002.csv', repeat = 10)
    #plot_accuracy("results/tanh_1002.csv", 'results/Plots/tanh2_100.png')
    #plot_accuracy_inv("results/tanh_1002.csv", 'results/Plots/tanh2_100_inv.png')

    # accuracy_comp_function(Brusselator_Data_Noise, N = 200, testsize = 1000, savefilename = 'results/Noise_Data.csv')
    # plot_accuracy("results/Noise_Data.csv", 'results/Plots/Noise_200.png')

    N = [20,30,40,50,60,70,80,100, 150,200,300, 500]
    # accuracy_incre_compare_func(Generate_data_tanh, N, "results/accuracy_sample_inc_tanh.csv", testsize = 1000, repeat = 20)
    # plot_accuracy_comp_inc("results/accuracy_sample_inc_tanh.csv", "results/Plots/accuracy_sample_inc_tanh.png",N)
    # plot_accuracy("results/Noise_Data.csv", 'results/Plots/Noise_200.png')
    #noise example
    # accuracy_comp_function(Brusselator_Data_Noise, N = 200, testsize = 1000, savefilename = 'results/Noise_Data.csv')
    #plot_accuracy("results/Noise_Data.csv", 'results/Plots/Noise_200.png')
    #plot_accuracy_inv("results/Noise_Data.csv", 'results/Plots/Noise_200_inv.png')




if __name__ == '__main__':
    main()
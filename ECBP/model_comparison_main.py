from model_utility import Generate_data_Curve, Euclidean_distance, Generate_data_Circle
from CBP import CBPClassifier, CBP, refrenced_method, GPSVM, GLSVM,LSVM
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import contextlib
import io,time
import seaborn as sns
from sklearn import preprocessing
from brusselator import Brusselator_Data

#models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.svm import SVC

BootStrapRepeat = 5

####Single classifiers:
def G_PSVM_Binary(X_train, Y_train):
    n = len(Y_train)
    #kernel coeficients
    K = [ 5,7, 9, 12]
    k = [1,3, 5]
    parameters = [[i, j] for i in K for j in k]
    accMatrix = np.zeros((len(parameters),BootStrapRepeat))
    for i in range(len(parameters)):
        for j in range(BootStrapRepeat):
            trainIndex = np.random.choice([k for k in range(n)],n)
            while len(np.unique(Y_train[trainIndex])) == 1:
                trainIndex = np.random.choice([k for k in range(n)],n)
            trainIndex  = np.unique(trainIndex)
            testIndex = [k for k in range(n) if k not in trainIndex]
            xTrain, yTrain = X_train[trainIndex], Y_train[trainIndex] 
            xTest, yTest = X_train[testIndex], Y_train[testIndex]
            model = GPSVM(xTrain, yTrain, parameters[i])
            predicts = model.predict(xTest)
            accMatrix[i,j] = np.sum(predicts == yTest)/len(yTest)
    # optimal K
    optimalPara = parameters[accMatrix.mean(axis=1).argsort()[-1]]
    #predict
    model = GPSVM(X_train, Y_train, optimalPara)
    return  model



def ECBP_PPR_base(X_train, Y_train):
    n = len(Y_train)
    #kernel coeficients
    r1 = [ 5, 10]
    degree1 = [3, 5]
    parameters = [[i, j, i, j] for i in r1 for j in degree1]
    accMatrix = np.zeros((len(parameters),BootStrapRepeat))
    for i in range(len(parameters)):
        for j in range(BootStrapRepeat):
            trainIndex = np.random.choice([k for k in range(n)],n)
            while len(np.unique(Y_train[trainIndex])) == 1:
                trainIndex = np.random.choice([k for k in range(n)],n)
            print(j)
            testIndex = [k for k in range(n) if k not in trainIndex]
            xTrain, yTrain = X_train[trainIndex], Y_train[trainIndex] 
            xTest, yTest = X_train[testIndex], Y_train[testIndex]
            model = CBPClassifier(xTrain, yTrain)
            #with contextlib.redirect_stdout(io.StringIO()):  
            model.fit(average = False, kind = 'PPR', args = parameters[i])
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
    for i in range(len(parameters)):
        for j in range(BootStrapRepeat):
            trainIndex = np.random.choice([k for k in range(n)],n)
            while len(np.unique(Y_train[trainIndex])) == 1:
                trainIndex = np.random.choice([k for k in range(n)],n)
            testIndex = [k for k in range(n) if k not in trainIndex]
            xTrain, yTrain = X_train[trainIndex], Y_train[trainIndex] 
            xTest, yTest = X_train[testIndex], Y_train[testIndex]
            model = refrenced_method(xTrain, yTrain, parameters[i][0], parameters[i][1])
            predicts = model.predict(xTest)
            accMatrix[i,j] = np.sum(predicts == yTest)/len(yTest)
    # optimal K
    optimalPara = parameters[accMatrix.mean(axis=1).argsort()[-1]]
    print(optimalPara)
    #predict
    model = refrenced_method(X_train, Y_train, optimalPara[0], optimalPara[1])
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
            while len(np.unique(Y_train[trainIndex])) == 1:
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
            while len(np.unique(Y_train[trainIndex])) == 1:
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
            while len(np.unique(Y_train[trainIndex])) == 1:
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
    n = len(Y_train)
    num_round = 5000
    eta = [0.01, 0.1,  0.3,  0.7]
    max_depth = [2,6,10,12]
    sub_sample = [0.3, 0.5, 1]
    parameters = [[i, j, k] for i in eta for j in max_depth for k in sub_sample]
    accMatrix = np.zeros((len(parameters),BootStrapRepeat))
    for i in range(len(parameters)):
        for j in range(BootStrapRepeat):
            trainIndex = np.random.choice([k for k in range(n)],n)
            while len(np.unique(Y_train[trainIndex])) == 1:
                trainIndex = np.random.choice([k for k in range(n)],n)
            testIndex = [k for k in range(n) if k not in trainIndex]
            xTrain, yTrain = X_train[trainIndex], Y_train[trainIndex] 
            xTest, yTest = X_train[testIndex], Y_train[testIndex]
            dTrain = xgb.DMatrix(xTrain, label = yTrain)
            dTest = xgb.DMatrix(xTest)
        
            # create train / validation set
            trainX, testX, trainY, testY = train_test_split(xTrain, yTrain, test_size=0.33, random_state=42)
            sub_dTrain = xgb.DMatrix(trainX, label = trainY)
            sub_dTest = xgb.DMatrix(testX, label = testY)
            evallist = [(sub_dTest,'eval'),(sub_dTrain,'train')]
            para = {'eta': parameters[i][0],'max_depth':parameters[i][1], 'subsample':parameters[i][2], 'nthread':4, 'eval_metric':'error' }
            with contextlib.redirect_stdout(io.StringIO()):       
                model = xgb.train(para, sub_dTrain, num_round, evallist, early_stopping_rounds = 1000)
            predicts = model.predict(dTest, iteration_range = (0,model.best_iteration+1))
            predicts[predicts >= 0.5] = 1
            predicts[predicts < 0.5] = 0
            accMatrix[i,j] = np.sum(predicts == yTest)/len(yTest)
    # optimal K
    optimalPara = parameters[accMatrix.mean(axis=1).argsort()[-1]]
    print(optimalPara)
    para = {'eta': optimalPara[0],'max_depth':optimalPara[1], 'subsample':optimalPara[2], 'nthread':4, 'eval_metric':'error'}
    #predict
    dTrain = xgb.DMatrix(X_train, label = Y_train)
    dTest = xgb.DMatrix(X_predict)

    #validation
    trainX, testX, trainY, testY = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)
    sub_dTrain = xgb.DMatrix(trainX, label = trainY)
    sub_dTest = xgb.DMatrix(testX, label = testY)
    evallist = [(sub_dTest,'eval'),(sub_dTrain,'train')]
    with contextlib.redirect_stdout(io.StringIO()): 
        model = xgb.train(para, sub_dTrain, num_round, evallist, early_stopping_rounds = 1000)
    predicts = model.predict(dTest, iteration_range = (0, model.best_iteration+1))
    predicts[predicts >= 0.5] = 1
    predicts[predicts < 0.5] = 0
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return  accuracy

def SVM_RBF(X_train, Y_train, X_predict, Y_predict):
    n = len(Y_train)
    #kernel coeficients
    gamma = [ 0.1, 0.5, 0.7,0.8,0.9,1,1.2, 1.5]
    parameters = [[i] for i in gamma]
    accMatrix = np.zeros((len(gamma),BootStrapRepeat))
    for i in range(len(gamma)):
        for j in range(BootStrapRepeat):
            trainIndex = np.random.choice([k for k in range(n)],n)
            while len(np.unique(Y_train[trainIndex])) == 1:
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

def SVM_poly(X_train, Y_train, X_predict, Y_predict):
    n = len(Y_train)
    #kernel coeficients
    gamma = [ 0.1, 0.5, 0.7,0.8,0.9,1,1.2, 1.5]
    degree = [1,2,3,4,5,6]
    cof0 = [0,0.1,0.3,0.5,0.7,1]
    parameters = [[i, j, k] for i in gamma for j in degree for k in cof0]
    accMatrix = np.zeros((len(gamma),BootStrapRepeat))
    for i in range(len(gamma)):
        for j in range(BootStrapRepeat):
            trainIndex = np.random.choice([k for k in range(n)],n)
            while len(np.unique(Y_train[trainIndex])) == 1:
                trainIndex = np.random.choice([k for k in range(n)],n)
            testIndex = [k for k in range(n) if k not in trainIndex]
            xTrain, yTrain = X_train[trainIndex], Y_train[trainIndex] 
            xTest, yTest = X_train[testIndex], Y_train[testIndex]
            model = SVC( gamma = parameters[i][0], degree = parameters[i][1], coef0 = parameters[i][2],kernel = 'poly').fit(xTrain,yTrain)
            predicts = model.predict(xTest)
            accMatrix[i,j] = np.sum(predicts == yTest)/len(yTest)
    # optimal K
    optimalPara = parameters[accMatrix.mean(axis=1).argsort()[-1]]
    print(optimalPara)
    #predict
    model = SVC(gamma = optimalPara[0], degree = optimalPara[1], coef0 = optimalPara[2],kernel = 'poly').fit(xTrain,yTrain)
    predicts = model.predict(X_predict)
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
    model = CBPClassifier(X_train, Y_train)
    model.fit(kind = '1dSpline', args = ['linear'])
    predicts = model.predict(X_predict)
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return  accuracy
def ECBP_avg(X_train, Y_train, X_predict, Y_predict):
    model = CBPClassifier(X_train, Y_train)
    model.fit(average = True, kind = '1dSpline', args = ['linear'])
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
    n = len(Y_train)
    #kernel coeficients
    K = [ 5,10,15,20]
    C = [0.1,0.5,0.75,1,1.25, 1.5]
    parameters = [[i, j] for i in K for j in C]
    accMatrix = np.zeros((len(parameters),BootStrapRepeat))
    for i in range(len(parameters)):
        for j in range(BootStrapRepeat):
            trainIndex = np.random.choice([k for k in range(n)],n)
            while len(np.unique(Y_train[trainIndex])) == 1:
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
    predicts = model.predict(X_predict)
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return  accuracy

def G_PSVM(X_train, Y_train, X_predict, Y_predict):
    model = multiclassClassifier(G_PSVM_Binary, X_train, Y_train)
    predicts = model.predict(X_predict)
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return  accuracy
def G_LSVM(X_train, Y_train, X_predict, Y_predict):
    n = len(Y_train)
    #kernel coeficients
    K = [ 3,6,10, 20]
    k = [1,3,5]
    parameters = [[i, j] for i in K for j in k]
    accMatrix = np.zeros((len(parameters),BootStrapRepeat))
    for i in range(len(parameters)):
        for j in range(BootStrapRepeat):
            trainIndex = np.random.choice([k for k in range(n)],n)
            while len(np.unique(Y_train[trainIndex])) == 1:
                trainIndex = np.random.choice([k for k in range(n)],n)
            trainIndex = np.unique(trainIndex)
            testIndex = [k for k in range(n) if k not in trainIndex]
            xTrain, yTrain = X_train[trainIndex], Y_train[trainIndex] 
            xTest, yTest = X_train[testIndex], Y_train[testIndex]
            model = GLSVM(xTrain, yTrain, parameters[i])
            predicts = model.predict(xTest)
            accMatrix[i,j] = np.sum(predicts == yTest)/len(yTest)
    # optimal K
    optimalPara = parameters[accMatrix.mean(axis=1).argsort()[-1]]
    print(optimalPara)
    #predict
    model = GLSVM(X_train, Y_train, optimalPara)
    predicts = model.predict(X_predict)
    accuracy = np.sum(predicts == Y_predict)/len(Y_predict)
    return  accuracy

def ECBP_Gradient(X_train, Y_train, X_predict, Y_predict):
    n = len(Y_train)
    #kernel coeficients
    K = [ 3,6,10]
    parameters = [[i, j] for i in K]
    accMatrix = np.zeros((len(parameters),BootStrapRepeat))
    for i in range(len(parameters)):
        for j in range(BootStrapRepeat):
            trainIndex = np.random.choice([k for k in range(n)],n)
            testIndex = [k for k in range(n) if k not in trainIndex]
            xTrain, yTrain = X_train[trainIndex], Y_train[trainIndex] 
            xTest, yTest = X_train[testIndex], Y_train[testIndex]
            model = ECBP(xTrain, yTrain, parameters[i])
            predicts = model.predict(xTest)
            accMatrix[i,j] = np.sum(predicts == yTest)/len(yTest)
    # optimal K
    optimalPara = parameters[accMatrix.mean(axis=1).argsort()[-1]]
    print(optimalPara)
    #predict
    model = GLSVM(X_train, Y_train, optimalPara)
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

def accuracy_incre_compare(N, testsize = 1000, repeat = 10):
    # KNN,rf, nnet, xgboost, SVM_RBF,SVM_poly,SVM_Linear,  ECBP, ECBP_avg, ECBP_PPR_avg, L_SVM, G_PSVM, G_LSVM, Referenced method
    # N train size
    names = ['KNN','rf', 'nnet', 'xgboost', 'SVM_RBF','SVM_poly','SVM_Linear',
      'ECBP', 'ECBP_avg', 'ECBP_PPR', 'L_SVM', 'G_PSVM', 'G_LSVM', 'Referenced method']
    classifier = [KNN, rf, nnet, xgBoost,SVM_RBF, SVM_poly,SVM_Linear,ECBP,ECBP_avg,ECBP_PPR,L_SVM,G_PSVM,G_LSVM,Refrenced_method]
    accu_matrix = np.zeros((len(N), len(names)))
    for i in range(len(N)):
        for r in range(repeat):
            print(i,'----',r)
            A = Generate_data_Circle(N[i])
            B = Generate_data_Circle(testsize)
            xTrain = A.iloc[:,0:2].values
            yTrain = A.iloc[:,2].values
            xTest = B.iloc[:,0:2].values
            yTest = B.iloc[:,2].values
            for j in range(len(classifier)):
                accu_matrix[i][j] = accu_matrix[i][j] + classifier[j](xTrain, yTrain, xTest,yTest)/repeat
    np.savetxt("results/accuracy_sample_circle_01.csv", accu_matrix, delimiter=",", header = '')
    return


def relative(col):
    return col/max(col)

def accuracy_comp(data,  savefilename, repeat = 10, folds = 5):
    # KNN,rf, nnet, xgboost, SVM_RBF,SVM_poly,SVM_Linear,  ECBP, ECBP_avg, ECBP_PPR_avg, L_SVM, G_PSVM, G_LSVM, Referenced method
    # 
    A = data
    N = len(A)
    names = ['KNN','rf', 'nnet', 'xgboost', 'SVM_RBF','SVM_poly','SVM_Linear',
       'ECBP_PPR', 'L_SVM', 'G_PSVM', 'Referenced method']
    indexes = []
    accuracy = np.zeros((repeat*folds, len(names)))
    classifier = [KNN, rf, nnet, xgBoost,SVM_RBF, SVM_poly,SVM_Linear,ECBP_PPR,L_SVM,G_PSVM,Refrenced_method]
    index = [int(N/folds)*i for i in range(folds)]
    index.append(N)
    numClass = len(np.unique(A.iloc[:,-1]))
    for i in range(5):
        indexes.append(np.arange(index[i], index[i+1]))
    for r in range(repeat):
        A = A.sample(frac=1)
        while len(np.unique(A.iloc[:,-1].values)) != numClass:
            A = A.sample(frac=1)
        A_train = A.iloc[:,0:-1].values
        C_train = A.iloc[:,-1].values
        #
        for i in range(folds):
            print(r,'----',i)
            IndexTest = indexes[i]
            IndexTrain = [k for k in range(N) if k not in IndexTest]
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

 ########################################  ########################################
 ######################################## ########################################
 ################################################################################
  #########################################plot functions

def plot_accuracy(filename):
    names = ['KNN','random forest', 'nnet', 'xgboost', 'SVM_RBF','SVM_poly','SVM_Linear'
    , 'ECBP_PPR_avg', 'L_SVM', 'G_PSVM', 'G_LSVM', 'Referenced method']
    df = pd.read_csv(filename, header = None, names = names)
    plt.subplot(1,2,1)
    df.boxplot(fontsize = 5,rot=30)
    plt.ylabel('Accuracy')
    plt.subplot(1,2,2)
    df = df.apply(relative)
    df.boxplot(fontsize = 5,rot=30)
    plt.show()

def plot_accuracy_comp_inc(filename):
    names = ['KNN','random forest', 'nnet', 'xgboost', 'SVM_RBF','SVM_poly','SVM_Linear',
    'ECBP', 'ECBP_avg', 'ECBP_PPR', 'L_SVM', 'G_PSVM', 'Referenced method']
    df = pd.read_csv(filename, header = None, names = names)
    df['n'] = [30,40,60,80,100,150,200,250,500]
    df = df[['KNN','random forest', 'nnet', 'xgboost', 'SVM_RBF','SVM_poly','SVM_Linear',
     'ECBP_PPR', 'L_SVM', 'G_PSVM', 'Referenced method', 'n']]
    df.plot(x = 'n',kind = 'line', style='.-')
    plt.ylim([0,1])

    plt.show()

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
    df = pd.read_csv('data/iris.data', header = None)
    df.iloc[:,:-1] = preprocessing.normalize(df.iloc[:,:-1].values)
    C_train = df.iloc[:,-1]   
    le = preprocessing.LabelEncoder()
    le.fit(df.iloc[:,-1] )
    df.iloc[:,-1]  = le.transform(df.iloc[:,-1] )
    return df







def main():

    #A = Iris_Data()
    #accuracy_comp(A, savefilename = 'results/Irise_Accuracy_data.csv')

    A = Brusselator_Data(300, 5)
    accuracy_comp(A, savefilename = 'results/Brusselator_Accuracy_data_300_5.csv')
    #A_train = A.iloc[:,:-1].values  # preprocessing.normalize(abalone.iloc[:,1:].values)
    # C_train = A.iloc[:,-1].values
    # X_train,  X_predict, Y_train, Y_predict = train_test_split(A_train, C_train, test_size=0.7)
    # plt.subplot(1,2,1)
    # plt.scatter(X_train[:,0],X_train[:,1], c = Y_train, cmap = 'turbo')
    # model = multiclassClassifier(Refrenced_method_base, A_train, C_train)
    # # pred = G_PSVM(X_train, Y_train, X_predict,Y_predict )
    # print('-----')
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
    #plot_accuracy("Iris_data_result.csv")

    # plt.show()
    # A = Generate_data_Curve(200)
    # accuracy_comp_data(data = A,  filename = 'Curve_200.csv', repeat = 10, folds = 5)
    #plot_accuracy_comp_inc("results/accuracy_sample_circle.csv")
    #accuracy_incre(N = (30,40,60,80,100,150,200,250,500))






if __name__ == '__main__':
    main()
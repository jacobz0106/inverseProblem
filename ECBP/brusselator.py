
import numpy as np
from scipy.optimize import fsolve
from model_utility  import Generate_data_Curve, Euclidean_distance, Euclidean_distance_vector, Generate_data_Curve_systematic, Generate_data_Curve_half_circle, createSystematic2D
from CBP import CBP, CBPClassifier
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import random


##solve the model
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


integral_vec = np.vectorize(integral)
A_x = [1,1.2]
A_y = [3,3.1]


def Brusselator_Data(n, sep = 10):
    X = np.random.uniform(0.7, 1.5, n)
    Y = np.random.uniform(2.75, 3.25, n)
    integral_vec = np.vectorize(integral)
    def integrals(lambda1, lambda3 = 1.65,T = 5,n = 100):
        #Riemann sum approxiamation left sided
        return (1/T)*np.sum(np.sum(function_y(lambda1[0], lambda1[1],lambda3,T,n),axis = 1)*T/n )
    df=pd.DataFrame(data={"X":X,"Y":Y})
    Z = df.apply(integrals, axis = 1)
    min_value = min(Z)
    max_value = max(Z)
    increment = (max_value - min_value)/sep
    df['Z'] = np.zeros(n)
    for i in range(sep):
        index = np.logical_and( Z >= min_value + increment*i,  Z <= min_value + increment*(i+1))
        df['Z'][index] = i
    return df


#### not used, validation purpose
def getProb(n,sep = 10, Ax= [1,1.2],Ay= [3,3.1], D_lower= 3.7, D_upper = 4.0):
    data = np.random.uniform(low=[0.7,2.75], high=[1.5,3.25], size=(n,2))
    sZ = integral_vec(data[:,0], data[:,1], 1.65,5,50)
    df = pd.DataFrame(data = {'X':data[:,0], 'Y':data[:,1], 'Z':sZ})
    L = np.repeat(0,n)
    L = np.logical_and(df['Z'] >=  D_lower , df['Z'] <=  D_upper)
    t = D_lower
    for i in range(1,sep + 1):
        dis = (D_upper - D_lower)/sep
        L[np.logical_and(df['Z'] >= t, df['Z'] <  t + dis)] = i
        t = t + dis
    df['Lable'] = L
    # check whether point is in the event A
    I = np.logical_and(np.logical_and(df['X'] >= Ax[0], df['X'] <= Ax[1]), np.logical_and(df['Y'] >= Ay[0], df['Y'] <= Ay[1]))
    prob = 0
    for i in np.unique(df['Lable'][I]):
        prob += np.sum(df['Lable'][I] == i) / np.sum(df['Lable'] == i)*(1/sep)
    return prob


def trunc(I,I_lower,I_upper,c):
    while 1:
        num = np.unique(I)
        for i in range(len(np.unique(I))):
            num[i] = np.sum(I == np.unique(I)[i])
        min_acc = np.where(num == num.min())[0][0]
        if num[min_acc] >= c:
            break
        else:
            merge(I,I_lower,I_upper,min_acc)

#merge small incidence
def merge(I,I_lower,I_upper,loc):
    unique = np.unique(I)
    
    if loc == 0:
        I_upper[I == unique[loc]] = I_upper[I == unique[loc+1]][0]
        I_lower[I==unique[loc+1]] = I_lower[I==unique[loc]][0]
        I[I==unique[loc]] = unique[loc + 1]
    elif loc== len(np.unique(I))-1:
        I_lower[I == unique[loc]] = I_lower[I == unique[loc-1]][0]
        I_upper[I==unique[loc-1]] = I_upper[I==unique[loc]][0]
        I[I==unique[loc]] = unique[loc-1]
    elif np.sum(I==unique[loc-1]) > np.sum(I==unique[loc+1]):
        I_upper[I == unique[loc]] = I_upper[I == unique[loc+1]][0]
        I_lower[I==unique[loc+1]] = I_lower[I==unique[loc]][0]
        I[I==unique[loc]] = unique[loc + 1]
    else:
        I_lower[I == unique[loc]] = I_lower[I == unique[loc-1]][0]
        I_upper[I==unique[loc-1]] = I_upper[I==unique[loc]][0]
        I[I==unique[loc]] = unique[loc-1]
   
## validation function, not used     
def Min_Sample(n, sep = 10, D_lower= 3.7, D_upper = 4.0):
    number = np.repeat(0.0,sep)
    for j in range(10):
        data = np.random.uniform(low=[0.7,2.75], high=[1.5,3.25], size=(n,2))
        sZ = integral_vec(data[:,0], data[:,1], 1.65,5,50)
        df = pd.DataFrame(data = {'X':data[:,0], 'Y':data[:,1], 'Z':sZ})
        L = np.repeat(0,n)
        L = np.logical_and(df['Z'] >=  D_lower , df['Z'] <=  D_upper)
        t = D_lower
        for i in range(1,sep + 1):
            dis = (D_upper - D_lower)/sep
            L[np.logical_and(df['Z'] >= t, df['Z'] <  t + dis)] = i
            number[i-1] += np.sum(np.logical_and(df['Z'] >= t, df['Z'] <  t + dis))/10
            t = t + dis
    return number

###
def Classify(df1, df2, I, method = 'regression', args = []):
    I2 = np.zeros(( len(np.unique(I)-1),len(df2) ))
    train = df1[['X','Y']]
    test = df2[['X','Y']]
    pair_label = I==np.unique(I)[0]
    for cla in range(len(np.unique(I))-1):
        pair_label = I <= np.unique(I)[cla]
        model = CBPClassifier(df1[['X','Y']].values, pair_label)
        model.fit(method, args)
        I2[cla] = model.predict(df2[['X','Y']].values)
    pred = np.repeat(0, len(df2))
    for i in range(len(np.unique(I))):
        pred[np.logical_and(I2[i] == False, pred == 0)] = np.unique(I)[i]
    return pred

def Cal_prob(df, I, I_upper, I_lower, Ax= [1,1.2],Ay= [3,3.1], D_lower = 3.7, D_upper = 4.0):
    prob = 0
    for i in range( len(np.unique(df['Z'])) ):
        if I_upper[I == np.unique(df['Z'])[i]][0] <= D_lower:
            prob = prob+0
        elif I_lower[I == np.unique(df['Z'])[i]][0] >= D_upper:
            prob = prob+0
        elif I_upper[I == np.unique(df['Z'])[i]][0] >= D_upper:
            prob_bin = np.min([D_upper - I_lower[I == np.unique(df['Z'])[i]][0],
                     D_upper - D_lower])/(D_upper - D_lower)
            total_num_points = np.sum(df['Z'] == np.unique(df['Z'])[i])
            target = df[df['Z'] == np.unique(df['Z'])[i]]
            points_in_rec = np.sum(np.logical_and(np.logical_and(target['X'] >= Ax[0], target['X'] <= Ax[1]), 
                           np.logical_and(target['Y'] >= Ay[0], target['Y'] <= Ay[1])))
            prob = prob + prob_bin* points_in_rec/total_num_points
        elif I_upper[I == np.unique(df['Z'])[i]][0] < D_upper:
            prob_bin = np.min([I_upper[I == np.unique(df['Z'])[i]][0] - I_lower[I == np.unique(df['Z'])[i]][0],
                     I_upper[I == np.unique(df['Z'])[i]][0] - D_lower ])/(D_upper - D_lower)
            total_num_points = np.sum(df['Z'] == np.unique(df['Z'])[i])
            target = df[df['Z'] == np.unique(df['Z'])[i]]
            points_in_rec = np.sum(np.logical_and(np.logical_and(target['X'] >= Ax[0], target['X'] <= Ax[1]), 
                           np.logical_and(target['Y'] >= Ay[0], target['Y'] <= Ay[1])))
            prob = prob + prob_bin* points_in_rec/total_num_points
    return prob


# Main function
def getProb_ML(n,s,sep = 10,c = 10, rep = 10, method = 'regression', sample_method = 'random', Ax= [1,1.2],Ay= [3,3.1], D_lower = 3.7, D_upper = 4.0, args = []):
    Prob_Exhaus = np.repeat(0.0,rep)
    prob_knn = np.repeat(0.0,rep)
    prob_model = np.repeat(0.0,rep)
    for j in range(rep):
        if sample_method == 'random':
            data = np.random.uniform(low=[0.7,2.75], high=[1.5,3.25], size=(n,2))
            dataTrain = np.random.uniform(low=[0.7,2.75], high=[1.5,3.25], size=(s,2))
        elif sample_method == 'systematic':
            data = np.random.uniform(low=[0.7,2.75], high=[1.5,3.25], size=(n,2))
            dataTrain = createSystematic2D(s,[0.7,1.5],[2.75,3.25])
        # method of exhaustion
        sZTrain = integral_vec(dataTrain[:,0], dataTrain[:,1], 1.65,5,50)
        df = pd.DataFrame(data = {'X':data[:,0], 'Y':data[:,1]})
        dfTrain = pd.DataFrame(data = {'X':dataTrain[:,0], 'Y':dataTrain[:,1], 'Z':sZTrain})
        L = np.repeat(0,n)
        L = np.logical_and(dfTrain['Z'] >=  D_lower , dfTrain['Z'] <=  D_upper)
        t = D_lower 
        for i in range(1,sep + 1):
            dis = (D_upper - D_lower)/sep
            L[np.logical_and(dfTrain['Z'] >= t, dfTrain['Z'] <  t + dis)] = i
            t = t + dis
        dfTrain['Lable'] = L
        # check whether point is in the event A
        I = np.logical_and(np.logical_and(dfTrain['X'] >= Ax[0], dfTrain['X'] <= Ax[1]), np.logical_and(dfTrain['Y'] >= Ay[0], dfTrain['Y'] <= Ay[1]))
        prob = 0
        for i in np.unique(dfTrain['Lable'][I]):
            prob += np.sum(dfTrain['Lable'][I] == i) / np.sum(dfTrain['Lable'] == i)*(1/sep)
        Prob_Exhaus[j] = prob
        
        
        
        pred = df
        df1 = dfTrain
        I = np.repeat(0, len(df1))
        I_lower = np.zeros(len(df1))
        I_upper = np.zeros(len(df1))
        #patition into sep classes
        intv = (4.1 - 3.2)/sep
        for i in range(1,sep+1):
            I[np.logical_and(df1['Z'] >=  3.2 + (i-1)*intv, df1['Z'] < 3.2 + i*intv )] =i
            I_lower[np.logical_and(df1['Z'] >=  3.2 + (i-1)*intv, df1['Z'] < 3.2 + i*intv )] = 3.2 + (i-1)*intv
            I_upper[np.logical_and(df1['Z'] >=  3.2 + (i-1)*intv, df1['Z'] < 3.2 + i*intv )] = 3.2 + i*intv
        trunc(I,I_lower,I_upper,c)
        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=1)
        neigh.fit(df1[['X','Y']], I)
    
        df2_knn = pd.DataFrame(data = {'X':pred['X'], 'Y':pred['Y'], 'Z':neigh.predict(pred)} )
        df2_model =  pd.DataFrame(data = {'X':pred['X'], 'Y':pred['Y'], 'Z':Classify(df1[['X','Y']],pred,I, method, args) } )
        df1['Z'] = I
        frame1 = [df1,df2_knn]
        frame2 = [df1,df2_model]
        df_knn = pd.concat(frame1, sort = True)
        df_model = pd.concat(frame2, sort = True)
    # X: 1.0-1.2 Y: 3.0-3.1
        prob_knn[j] = Cal_prob(df_knn, I, I_upper, I_lower)
        prob_model[j] = Cal_prob(df_model, I, I_upper, I_lower)
        print(j)
    prob = np.zeros(rep*3)
    prob[0:rep] = Prob_Exhaus
    prob[rep:(rep*2)] = prob_knn
    prob[rep*2:rep*3] = prob_model
    return prob





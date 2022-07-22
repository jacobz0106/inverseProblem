from brusselator import getProb_ML, integral_vec, dQ_dlambda, Brusselator_Data, getProb
from model_utility import Generate_data_Curve, Euclidean_distance, Generate_data_Circle
from CBP import CBPClassifier, CBP, refrenced_method, GPSVM, LSVM
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
import matplotlib.cm as cm
import pandas as pd
import random
import seaborn as sns
from pathlib import Path
from scipy import optimize 
import matplotlib.colors as colors
from sklearn import cluster
from random import randint
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions

from scipy.optimize import basinhopping
from matplotlib.patches import FancyArrowPatch, ArrowStyle
from sklearn.neighbors import KNeighborsClassifier
import time
from sklearn.svm import SVC

""" Example problem
    https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.optimize.basinhopping.html
"""


""" Example bounds """


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


def density_plot(k  = 0):
    m = 10
    width = 0.8/m
    height = 0.5/m
    x, y = np.mgrid[0.7:1.5+width:width, 2.75:3.25+height:height]
    @np.vectorize
    def getProb_vec(a, b, n = 1000, sep = 20):
        print([a,b])
        return getProb(n,sep = 20, Ax= [a,a+width],Ay= [b,b+height], D_lower= 3.7, D_upper = 4.0)
    z0 = getProb_vec(x,y, 100, 10)
    z1 = getProb_vec(x,y, 200, 10)
    z2 = getProb_vec(x,y, 400)
    z3 = getProb_vec(x,y, 800)
    z0 = z0/(width*height)
    #z_min, z_max = np.array([z0,z1,z2,z3]).min(), np.array([z0,z1,z2,z3]).max()
    # z1 = z1[:-1, :-1]/(width*height)
    # z2 = z2[:-1, :-1]/(width*height)
    # z3 = z3[:-1, :-1]/(width*height)
    # z_min, z_max = np.array([z0,z1,z2,z3]).min(), np.array([z0,z1,z2,z3]).max()

    fig, axs = plt.subplots(2, 2)

    ax = axs[0, 0]
    c = ax.contourf(x, y, z0, cmap='plasma')
    ax.set_title('100 points/10 bins')
    fig.colorbar(c, ax=ax)

    # ax = axs[0, 1]
    # c = ax.pcolormesh(x, y, z1, cmap='plasma', vmin=z_min, vmax=z_max)
    # ax.set_title('200 points/10 bins')
    # fig.colorbar(c, ax=ax)

    # ax = axs[1, 0]
    # c = ax.pcolormesh(x, y, z2, cmap='plasma', vmin=z_min, vmax=z_max)
    # ax.set_title('400 points/20 bins')
    # fig.colorbar(c, ax=ax)

    # ax = axs[1, 1]
    # c = ax.pcolormesh(x, y, z3, cmap='plasma', vmin=z_min, vmax=z_max)
    # ax.set_title('800 points/20 bins')
    # fig.colorbar(c, ax=ax)
    # # plt.show()
    plt.savefig(Path('results/Brusselator_result_density_Exhau_5.png'))

    # m = 25
    # width = 0.8/m
    # height = 0.5/m

    # @np.vectorize
    # def getProb_vec(a, b):
    #   P = getProb_ML(n = 1000,s = 400,sep = 20,c = 10, rep = 1, method = 'gradient',sample_method = "random")
    #   return (P[0],P[1], P[2])
    # # generate 2 2d grids for the x & y bounds
    # x, y = np.mgrid[0.7:1.5+width:width, 2.75:3.25+height:height]
    # z0,z1,z2 = getProb_vec(x,y)
    # # x and y are bounds, so z should be the value *inside* those bounds.
    # # Therefore, remove the last value from the z array.
    # z0,z1,z2 = z0[:-1, :-1], z1[:-1, :-1], z2[:-1, :-1]
    # z_min, z_max = np.array([z0,z1,z2]).min(), np.array([z0,z1,z2]).max()

    # fig, axs = plt.subplots(2, 2)

    # ax = axs[0, 0]
    # c = ax.pcolor(x, y, z0, cmap='plasma', vmin=z_min, vmax=z_max)
    # ax.set_title('Exhausive with 400 points and 20 bins')
    # fig.colorbar(c, ax=ax)

    # ax = axs[0, 1]
    # c = ax.pcolormesh(x, y, z1, cmap='plasma', vmin=z_min, vmax=z_max)
    # ax.set_title('knn with 400 points(1000 additional trained) and 20 bins')
    # fig.colorbar(c, ax=ax)


    # ax = axs[1, 0]
    # c = ax.pcolorfast(x, y, z3, cmap='plasma', vmin=z_min, vmax=z_max)
    # ax.set_title('1dSpline with 400 points(1000 additional trained) and 20 bins')
    # fig.colorbar(c, ax=ax)

    # fig.tight_layout()
    # plt.show()

def density_plt(n = 200, width = 0.05):
    m = 20
    w = 0.8/m
    h = 0.5/m
    data = np.random.uniform(low=[0.7,2.75], high=[1.5,3.25], size=(n,2))
    sZ = integral_vec(data[:,0], data[:,1], 1.65,5,50)
    df = pd.DataFrame(data = {'X':data[:,0], 'Y':data[:,1], 'Z':sZ})
    x, y = np.mgrid[0.7:1.5:w, 2.75:3.25:h]
    @np.vectorize
    def getProb_vec(a, b, width):
        print([a,b])
        I = integral_vec(a, b, 1.65,5,50)
        sum_ = np.sum(np.logical_and(np.logical_and(df['Z'] >= I - width/2, df['X'] <= I + width/2),
            np.logical_and(df['Z'] >= 3.7, df['X'] <= 4.0)
            ))
        total = np.sum(np.logical_and(df['Z'] >= 3.7, df['X'] <= 4.0))
        return sum_/total
    z = getProb_vec(x,y, width)
    fig, axs = plt.subplots(2, 2)

    ax = axs[0, 0]
    c = ax.contourf(x, y, z, cmap='plasma')
    ax.set_title('1000 points')
    fig.colorbar(c, ax=ax)
    plt.show()


def boundary_plot(N, typ, seed, args = []):
    np.random.seed(seed)
    A = Generate_data_Curve(N)
    A_train = A.iloc[:,0:2].values
    C_train = A.iloc[:,2].values
    model = CBPClassifier(A_train, C_train)
    print(args)
    model.fit(typ, args)
    points_posi = np.unique(np.array(model.cbp.points).reshape(-1,))
    boundary_candi = A.iloc[points_posi,:]
    plt.subplot(1, 2, 1)
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.scatter(A['X'], A['Y'], c=A['Z'], cmap= 'rainbow', s = 10)
    plt.plot(np.arange(0,10,0.1), np.sin(np.arange(0,10,0.1)) +5, linestyle='-.')
    plt.scatter(boundary_candi['X'],boundary_candi['Y'], c='none', edgecolors='green', 
                cmap = 'rainbow', marker = 'H', s = 100, label='Gabriel edited set')
    # for i in model.cbp.points:
    #     plt.plot(A.iloc[i,0],A.iloc[i,1], linestyle='-.', color = 'red')
    #draw boundaries
    plt.plot(np.arange(0,10,0.1),
          model.fit_upper(np.arange(0,10,0.1)), linestyle='-')
    plt.plot(np.arange(0,10,0.1),
          model.fit_lower(np.arange(0,10,0.1)), linestyle='-')
    #plt.plot(np.arange(0,10,0.1), (model.fit_upper(np.arange(0,10,0.1)) + model.fit_lower(np.arange(0,10,0.1)) )/2, linestyle='-.', c = 'red')
    ax = plt.subplot(1, 2, 2)
    ax.plot(np.arange(0,10,0.1), np.sin(np.arange(0,10,0.1)) +5, linestyle='-.')
    #plt.scatter(A['X'], A['Y'], c=model.predict(A_train), cmap= 'rainbow', s = 10)
    plt.xlim([0,10])
    plt.ylim([0,10])
    kmeans = cluster.KMeans(n_clusters=4, random_state=0).fit(np.array(model.cbp.midpoints))
    scatter = ax.scatter(np.array(model.cbp.midpoints)[:,0], np.array(model.cbp.midpoints)[:,1],s = 30, 
        c = kmeans.labels_, cmap ='turbo', marker = '+')
    legend1 = ax.legend(*scatter.legend_elements(), title = 'cluster')
    ax.scatter(np.array(kmeans.cluster_centers_)[:,0], np.array(kmeans.cluster_centers_)[:,1], s = 50, c = np.unique(kmeans.labels_), cmap = 'turbo')
    ax.add_artist(legend1)
    plt.show()

def boundary_plot_LSVM(N, seed, args = []):
    np.random.seed(seed)
    A = Generate_data_Curve(N)
    A_train = A.iloc[:,0:2].values
    C_train = A.iloc[:,2].values
    model = LSVM(A_train, C_train, args)
    plt.xlim([0,10])
    plt.ylim([0,10])
    plt.scatter(A['X'], A['Y'], c=model.predict(A_train), cmap= 'rainbow', s = 10)
    plt.plot(np.arange(0,10,0.1), np.sin(np.arange(0,10,0.1)) +5, linestyle='-.')
    plt.show()

def boundary_GPSVM(seed, N, args = []):
    np.random.seed(seed)
    #A = Generate_data_Curve(N)
    A = Generate_data_Circle(N)
    A_train = A.iloc[:,0:2].values
    C_train = A.iloc[:,2].values
    model = GPSVM(A_train, C_train, args)
    #SVM_model = SVC(gamma = 0.1,kernel = 'rbf').fit(A_train,C_train)
    #SVM_model = SVC(degree = 4,gamma = 1,kernel = 'poly').fit(A_train,C_train)
    points_posi = np.unique(np.array(model.cbp.points).reshape(-1,))
    boundary_candi = A.iloc[points_posi,:]
    fig, ax = plt.subplots()
    #boundary
    #ax.plot(np.arange(0,10,0.1), np.sin(np.arange(0,10,0.1)) +5, linestyle='-.', c = 'purple', label = 'true boundary')
    kmeans = cluster.KMeans(n_clusters=args[0], random_state=0).fit(np.array(model.cbp.midpoints))


    #----scatter plot-------
    # label0 = C_train == 0
    # label1 = C_train == 1
    # plt.xlim([0,10])
    # plt.ylim([0,10])
    # circle = plt.Circle((5,5), 3, fill = False, linestyle = '-.', edgecolor = 'purple', label = 'true boundary')
    # ax.add_artist(circle)
    # pred = model.predict(A_train)
    # plt.scatter(A['X'][label0], A['Y'][label0], marker = '+',s = 30, c = 'black')
    # plt.scatter(A['X'][label1], A['Y'][label1], marker = '_',s = 30, c = 'black')

    #-----------------------
    # plt.xlim([0,10])
    # plt.ylim([0,10])
    # circle = plt.Circle((5,5), 3, fill = False, linestyle = '-.', edgecolor = 'purple', label = 'true boundary')
    # ax.add_artist(circle)
    # pred = model.predict(A_train)
    # label0 = pred == 0
    # label1 = pred == 1
    # plt.scatter(A['X'][label0], A['Y'][label0], marker = '+',s = 30, c = 'black')
    # plt.scatter(A['X'][label1], A['Y'][label1], marker = '_',s = 30, c = 'black')
    plot_decision_regions(A_train, C_train, clf = model, legend = 2)
    plt.gcf().set_size_inches(10,10)
    plt.savefig(Path('results/Diagram_Decison_boundary_3.png'), bbox_inches='tight')
    return


    #plt.scatter(SVM_model.support_vectors_[:,0], SVM_model.support_vectors_[:,1], marker = '8',facecolors="None", edgecolors= 'red', s = 60, label = 'support_vectors' )
    # label0 = boundary_candi['Z'] == 0
    # label1 = boundary_candi['Z'] == 1
    # plt.scatter(boundary_candi['X'][label0], boundary_candi['Y'][label0],  label = 'Gabriel edited set +', marker = '+', s = 30, c = 'black')
    # plt.scatter(boundary_candi['X'][label1], boundary_candi['Y'][label1],  label = 'Gabriel edited set -', marker = '_', s = 30, c = 'black')
    # plt.scatter(model.cbp.midpoints[:,0], model.cbp.midpoints[:,1],  label = 'CBPs', marker = '.', s = 30, c = 'black')
    # ct = ax.scatter(np.array(kmeans.cluster_centers_)[:,0], np.array(kmeans.cluster_centers_)[:,1], 
    #     s = 30, c = np.unique(kmeans.labels_), cmap = 'turbo', label = 'cluster center', marker = 's')
    # ind = np.array([])
    # label = np.array([])
    # for i in range(model.clusterNum):
    #     midpoints_subset_index = model.kmeans.labels_ == model.clusterLabel[i]
    #     Gabriel_pairs = np.array(model.cbp.points)[midpoints_subset_index]
    #     subset_index = Gabriel_pairs.reshape(-1)
    #     subset_label = np.repeat(i, len(subset_index))
    #     label = np.concatenate((label,subset_label), axis=None)
    #     ind= np.concatenate((ind,subset_index), axis=None)

    # label = label.astype(int)/model.clusterNum
    # ind = ind.astype(int)
    # c = plt.cm.turbo(label)
    # a = plt.scatter(A_train[ind][:,0],A_train[ind][:,1], facecolors="None", edgecolors= c, marker = 'H', s = 100, label='Clustered Points')
    
    # # straight lines
    # for i in range(model.clusterNum):
    #     center = model.clusterCentroids[i][0]

    #     w = model.SVM[i].coef_[0]           # w consists of 2 elements
    #     b = model.SVM[i].intercept_[0]      # b consists of 1 element  \
    #     x = np.linspace(center - 0.8, 
    #         center+ 0.8, 20)
    #     y_points = -(w[0] / w[1]) * x - b / w[1]  # getting corresponding y-points
    #     index = (y_points - model.clusterCentroids[i][1])**2 + (x - center)**2 < 4
    #     plt.plot(x[index], y_points[index], c = plt.cm.turbo(i/model.clusterNum), linewidth = 2)

    plt.legend(loc='upper right',facecolor='white', framealpha=1, frameon = True)
    plt.gcf().set_size_inches(10,10)
    plt.savefig(Path('results/Diagram_plot6.png'), bbox_inches='tight')
    return


def time_Comp(N, k, repeat = 10):
    time1 = np.zeros((len(N), k))
    time2 = np.zeros((len(N), k))
    for i in range(len(N)):
        for j in range(repeat):
            A = Generate_data_Curve(N[i])
            A_train = A.iloc[:,0:2].values
            C_train = A.iloc[:,2].values
            start =  time.time()

            stop =  time.time()
            time1[i,j] = stop - start

            start =  time.time()
            cbp = CBP(A_train,C_train)
            stop =  time.time()
            time2[i,j] = stop - start
    df = pd.DataFrame( {'Kmeans':time1[:,:repeat].reshape(-1), 'CBP+Kmeans':time2[:,:repeat].reshape(-1),
                    'n':np.repeat(N, repeat)} )
    df_plot = df.melt(id_vars='n', value_vars=['Kmeans', "CBP+Kmeans"])
    graph = sns.boxplot(x='n', y='value', hue='variable', data=df_plot)
    plt.show()
    return


def partial(N, folds):
    A = Generate_data_Curve(N)
    A_train = A.iloc[:,0:2].values
    C_train = A.iloc[:,2].values
    cbp = CBP(A_train,C_train)
    Gabriel_editedSet = np.unique(np.array(cbp.points).reshape(-1))
    plt.subplot(1, 3, 1)
    plt.scatter(A_train[Gabriel_editedSet][0], A_train[Gabriel_editedSet][1], C_train[Gabriel_editedSet][0], cmap= 'rainbow', s = 10)

    index = [int(N/folds)*i for i in range(folds)]
    Train = []        
    Label = []
    plt.subplot(1, 3, 2)
    for i in range(folds-1):
        Train = Train.append(A_train[index[i]:index[i+1]])
        Label = label.append(Label[index[i]:index[i+1]])
    Train = Train.append(A_train[index[folds]:])
    Label = label.append(Label[index[folds]:])

    plt.subplot(1, 3, 3)
    return




def main():
    #boundary_plot(200, 'PPR', seed = 111, args = [20, 6, 20, 6])
    #boundary_plot(500, 'MARS', seed = 111, args = [10,6,0.5,10,6,0.3])
    #boundary_plot(200, '1dSpline', seed = 111,args = ['cubic', 'cubic'])
    #
    boundary_GPSVM(seed = 99, N = 1000,args = [8,3])
    #SVM - KNN
    #boundary_plot_LSVM(seed = 2022, N = 5000,args = [20,0.5])
    #time_Comp([100,200,300,400,500,600,700,1000],k = 10,repeat = 10)
    # for N in [200,400,800,1600,3200,6400]:
    #     A = Generate_data_Curve(N)
    #     A_train = A.iloc[:,0:2].values
    #     C_train = A.iloc[:,2].values
    #     start =  time.time()
    #     cbp = CBP(A_train,C_train)
    #     stop =  time.time()
    #     print(N, stop - start, cbp.count)
    










if __name__ == '__main__':
    main()
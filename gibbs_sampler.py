import numpy as np
import pandas as pd
import random
import numpy.linalg as LA
from scipy.stats import multinomial
from scipy.stats import multivariate_normal
from scipy.stats import bernoulli
from scipy.stats import mode
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn import metrics
import time
import matplotlib.pyplot as plt


class gibbs_state:
    '''This class keeps track of the current state of the algorithm'''
  
    def __init__(self, labels, cluster_members, centers, time):
        self.labels = labels
        self.cluster_members = cluster_members
        self.centers = centers
        self.time = time

def gibbs_sampler(data, cov, seed=1, init_cluster_num=10, B=100):
    '''Runs Gibbs sampler for normally clustered data with constant covariance'''
    '''Returns predicted labels, labels for all states of algo, and cluster centers for all states of algo'''
    '''Returns total running time'''

    
    init_time = time.time()

    random.seed(seed)

    ################################
    # Initialize state
    ################################

    N = data.shape[0]
    d = data.shape[1]
    alpha = 1

    # Initiate random assignment of clusters
    clusters = list(range(init_cluster_num))
    init_labels = random.choices(clusters, k = N)

    # Save cluster members
    cluster_members = {}
    for c in clusters:
        cluster_members[c] = []
    for i in range(N):
        cluster_members[init_labels[i]].append(i)

    # Initialize cluster means with empirical average
    # Initialize cluster covariances with empirical covariances
    cluster_means = {}
    cluster_variances = {}
    for c in clusters:
        cluster_data = data[cluster_members[c],]
        cluster_means[c] = np.mean(cluster_data,axis=0)
        cluster_variances[c] = np.cov(cluster_data,rowvar=False)

    # Initialize state
    state = gibbs_state(labels=init_labels,
                     cluster_members=cluster_members,
                     centers=cluster_means,
                     time = init_time)
    

    ################################
    # Set Priors
    ################################

    mu0 = np.mean(data,axis=0) 
    Sigma0 = np.cov(data,rowvar=False) #multivariate normal prior on cluster means
    Sigma = cov*np.eye(d) #prior on cluster variances
    inv_Sigma = LA.inv(Sigma); inv_Sigma0 = LA.inv(Sigma0) #compute inverses for use later
    det_Sigma = LA.det(Sigma); det_Sigma0 = LA.det(Sigma0) #compute determinants for use later
    const_1 = (2*np.pi)**(-d/2)*det_Sigma**(-1/2)*det_Sigma0**(-1/2) #compute constant for Gibbs Sampler
    const_2 = LA.det(LA.inv(inv_Sigma0 - inv_Sigma))**(1/2) #compute constant for Gibbs Sampler

    ################################
    # Initialize cluster labels and state
    ################################


    #array to keep track of observation labels for each iteration
    labels_array = np.empty((B,N))
    labels_array[0] = init_labels

    #list to keep track of cluster centers at each iteration
    centers_list = []
    centers_list.append(state.centers)

    ################################
    # Run Gibbs Sampler
    ################################

    for b in range(1,B):

        ################################
        # Propose new cluster c_i_star
        ################################

        for i in range(N):

            #Get data point and current cluster
            x_i = data[i,]
            c_i = state.labels[i]

            #Get current clusters and number of cluster members
            clusters = list(state.cluster_members.keys())
            non_empty_clusters = list(state.centers.keys())
            clusters = [cluster for cluster in clusters if cluster in non_empty_clusters]
            number_in_cluster = [len(state.cluster_members[cluster]) - 1 if cluster == c_i else len(state.cluster_members[cluster]) for cluster in clusters]

            #Calculate constant
            square1 = np.dot(mu0,np.dot(inv_Sigma0,mu0))
            square2 = np.dot(x_i,np.dot(inv_Sigma,x_i))
            const_3 = np.exp(0.5*square1 - 0.5*square2)
            const = const_1*const_3/const_2

            #Calculate probabilities of joining new existing or new cluster
            p_joins_existing_cluster = [number_in_cluster[i]/(N - 1 + alpha)*multivariate_normal.pdf(x=x_i,mean=state.centers[clusters[i]],cov=Sigma) for i in range(len(clusters))]
            p_joins_new_cluster = alpha/(N - 1 + alpha)*const
            cluster_probs = p_joins_existing_cluster + [p_joins_new_cluster]
            sum_probs = sum(cluster_probs)
            cluster_probs = [prob/sum_probs for prob in cluster_probs]

            #Make list of cluster possibilities including all existing cluster and a possible new one
            cluster_possibilities = clusters + [clusters[-1] + 1]
            
            #Choose a new cluster for x_i
            c_i_star = np.random.choice(cluster_possibilities, size=1, p=cluster_probs)[0]

            #Determine whether c_i_star is a new cluster or not
            new_cluster = False
            if c_i_star == cluster_possibilities[-1]:
                new_cluster = True

            #If it is a new cluster, update the state and draw a new cluster center
            if new_cluster:
                state.labels[i] = c_i_star #change label
                state.cluster_members[c_i].remove(i) #remove from old cluster
                state.cluster_members[c_i_star] = [i] #add to new cluster
                state.centers[c_i_star] = multivariate_normal.rvs(mu0,Sigma0,size=1) #assign mean to the cluster
            #Otherwise, update existing cluster assignments
            else:
                state.labels[i] = c_i_star #change label
                state.cluster_members[c_i].remove(i) #remove from old cluster
                state.cluster_members[c_i_star].append(i) #add to new cluster
        
        ################################
        # Update cluster centers
        ################################

        #For each current cluster
        for c in list(state.cluster_members.keys()):
        
            #If there are no data points in the cluster, remove it and skip
            if len(state.cluster_members[c]) == 0:
                state.centers.pop(c,None)
                continue

            #Else, draw from posterior of cluster center
            cluster_members = state.cluster_members[c]
            num_c = len(cluster_members)
            cluster_member_data = data[cluster_members,]
            x_bar = np.mean(cluster_member_data,axis=0)
            S = LA.inv(inv_Sigma0 + num_c*inv_Sigma)
            m_second_term = np.dot(inv_Sigma0,mu0) + num_c*np.dot(inv_Sigma,x_bar)
            m = np.dot(S,m_second_term)
            state.centers[c] = multivariate_normal.rvs(m,S,size=1)

        ################################
        # Save labels, centers, and time
        # for this iteration
        ################################       

        #print(state.labels)
        labels_array[b] = state.labels
        centers_list.append(state.centers.copy())
        state.time = time.time()

    #Predict cluster labels from the mode of the labels array
    pred_y = mode(labels_array)[0]

    #Return prediction, labels from all iters, centers from all iters, and total running time
    return pred_y, labels_array, centers_list, state.time - init_time

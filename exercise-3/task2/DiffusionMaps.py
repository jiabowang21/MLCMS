import numpy as np
import scipy as sc
import math


def Diffusion_Map(data, L):
    """
    Applies the Diffusion Map algorithm to data to obtain L + 1 eigenfunctions.
    The algorithm maps data points into the set of eigenfunctions of the Laplace- Beltrami operator on a manifold describing the 
    data

    Parameters:
        data: the data where we want to apply the function 
        L: L + 1 is the number of eigenfunctions (eigenvectors and eigenvalues) we will obtain 
       
    Output:
        phil: eigenvector found after appliying the algorithm to the input data
        alphal: eigenvalue found after appliying the algorithm to the input data
    """
    
    #step 1 -> Form a distance matrix D with entries Dij = ||yi - yj|| with the sparse version
    kdtree = sc.spatial.KDTree(data)
    D = kdtree.sparse_distance_matrix(kdtree, 200).toarray()
    #D = euclidean_distances(data, data)
    
    
    #step 2 -> Set ε to 5% of the diameter of the dataset
    epsilon = 0.05 * np.max(D)
    #print(epsilon)
    
    #step 3 -> Form the kernel matrix W with Wij = exp(−(Dij)^2/ε)
    W = np.exp(-np.power(D, 2) / epsilon)
    #print(W)
    
    #step 4 -> Form the diagonal normalization matrix Pii = Sum Wij
    Sum_W_i = np.sum(W, axis = 0)
    #print(Sum_W_i)
    P = np.diag(Sum_W_i) 
    #print(P)
    
    #step 5 -> Normalize W to form the kernel matrix K = P−1 W P−1
    P_inv = np.linalg.inv(P)
    #print(P_inv)
    K = np.matmul(np.matmul(P_inv, W), P_inv)
    #print(K)
    
    #step 6 -> Form the diagonal normalization matrix Qii = Sum Kij
    Sum_K_i = np.sum(K, axis = 0)
    Q = np.diag(Sum_K_i)
    
    #step 7 -> Form the symmetric matrix Tˆ = Q^−1/2 K Q^−1/2.
    Q_inv = np.diag(1 / np.sqrt(Sum_K_i))
    T = np.matmul(np.matmul(Q_inv, K), Q_inv)
    
    #step 8 -> Find the L + 1 largest eigenvalues al and associated eigenvectors vl of Tˆ
    L += 1
    evalues, evectors = np.linalg.eigh(T)
    evalues = evalues[-L:][::-1]
    evector = evectors[:, -L:][:, ::-1]
    #print(evalues)
    #print(evector.size)
    
    #step 9 -> Compute the eigenvalues of Tˆ^1/ε by λl2 = al^1/ε
    #Note that the eigenvector φ0 is constant if the data set is connected for the given value of ε, 
    #so only the eigenvectors φ1 , φ2 , . . . are of interest.
    alphal = np.sqrt(np.power(evalues, (1/epsilon)))
    
    #step 10 -> Compute the eigenvectors φl of the matrix T = Q−1 K by φl = Q^−1/2vl
    phil = np.matmul(Q_inv, evector)
    
    return phil, alphal
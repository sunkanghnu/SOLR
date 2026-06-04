import numpy as np
from sklearn.metrics import pairwise_distances

def eca_bs(X_2D, distance_matrix = None, sigma = None):    
    """
    Implement the ECA (Entropy Clustering Algorithm) band selection algorithm
    Parameters:
        X_2D : Hyperspectral data with shape (N, L), where N is the number of samples and L is the number of bands
        distance_matrix : Precomputed distance matrix (optional). If None, it will be calculated automatically
        sigma : Kernel width parameter (optional). If None, it will be set adaptively
    Returns:
        sorted_indices : Indices of bands sorted by ES values in descending order
        ES : ES (Evaluation Score) values for all bands
    """
    # 1. Calculate distance matrix
    if distance_matrix is None:
        distance_matrix = pairwise_distances(X_2D.T, metric = 'euclidean')
    D = distance_matrix
    # 2. Adaptive parameter setting
    if sigma is None:     
        sigma = np.sqrt(np.mean(D) / 30)
    # 3. Calculate local density ρ
    rho = np.sum(np.exp(-D / (2 * sigma**2)), axis = 1)
    # 4. Calculate minimum distance δ
    delta = np.zeros_like(rho)
    max_density_idx = np.argmax(rho)
    
    for i in range(len(rho)):
        if i == max_density_idx:
            delta[i] = np.max(D[i])
        else:
            higher_density_idxs = np.where(rho > rho[i])[0]
            delta[i] = np.min(D[i, higher_density_idxs]) if len(higher_density_idxs) > 0 else 0
    
    # 5. Calculate ES values
    ES = rho * delta
    
    # 6. Band selection - sort indices by ES values in descending order
    sorted_indices = np.argsort(ES)[::-1]
    
    return sorted_indices, ES
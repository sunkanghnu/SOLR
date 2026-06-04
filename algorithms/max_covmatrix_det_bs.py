import numpy as np
import math
import warnings
# Ignore runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def max_covmatrix_det_bs(X_2D, num_selected_bands, covariance_matrix = None):
    """
    Performs feature selection using the Maximum Determinant Covariance Matrix (MDCM) algorithm.
    This method iteratively selects a subset of features that maximize the product of singular values (akin to the determinant)
    of the covariance submatrix aligning to the chosen features. The approach prefer subsets whose corresponding covariance 
    submatrix is well-conditioned in terms of spread of data.
    Args:
        X_2D (numpy.ndarray): 2D data matrix of shape (n_samples, n_features).
        num_selected_bands (int): Number of features to select.
        covariance_matrix (numpy.ndarray, optional): Covariance matrix of shape (n_features, n_features). 
            If None, it will be computed from X.
    Returns:
        numpy.ndarray: Indices of selected features (length = num_selected_bands).
    """
    _, num_bands = X_2D.shape
    if(covariance_matrix is None):
        covariance_matrix = np.cov(X_2D, rowvar = False)
    np.random.seed(42)
    selected_indices = np.sort(np.random.choice(num_bands, size = num_selected_bands, replace = False))        
    iter_times = 10
    for iter in range(iter_times):
        sub_conv_matrix = covariance_matrix[np.ix_(selected_indices, selected_indices)]
        s = np.linalg.det(sub_conv_matrix)
        err_old = s
        err_new = err_old
        for i in range(num_selected_bands):
            for j in range(num_bands):
                if (np.isin(j, selected_indices)):
                    continue
                selected_indices_tmp = selected_indices.copy()
                selected_indices_tmp[i] = j
                sub_conv_matrix = covariance_matrix[np.ix_(selected_indices_tmp, selected_indices_tmp)]
                s = np.linalg.det(sub_conv_matrix)
                err_tmp = s
                if(err_tmp > err_new):
                    err_new = err_tmp
                    selected_indices[i] = j
        if(np.abs(err_new - err_old)/np.abs(err_old + 1e-10) < 1e-6):
            break
    return selected_indices
    """
    Selects one band from each subset in band_subsets such that the determinant of the covariance matrix formed by the selected bands is maximized.
    Constraint: One band must be selected from each subset in band_subsets, and the total number of selected bands is num_selected_bands (the length of band_subsets must equal num_selected_bands).

    Args:
        X_2D (numpy.ndarray): 2D data matrix with shape (n_samples, n_features)
        num_selected_bands (int): Total number of bands to select (must equal the length of band_subsets)
        band_subsets (list/numpy.ndarray): List of band subsets, where each element is a subset (array/list) containing several band indices.
                                          The length must equal num_selected_bands, representing the optional band range for each position
        covariance_matrix (numpy.ndarray, optional): Precomputed covariance matrix with shape (n_features, n_features).
                                                    If None, it will be computed from X_2D.

    Returns:
        numpy.ndarray: Indices of selected bands (length=num_selected_bands), satisfying the condition of selecting one from each subset with maximum determinant
    """
    # Input validity check
    if len(band_subsets) != num_selected_bands:
        raise ValueError(f"The length of band_subsets ({len(band_subsets)}) must equal the number of bands to select num_selected_bands ({num_selected_bands})")
    for idx, subset in enumerate(band_subsets):
        if len(subset) == 0:
            raise ValueError(f"The {idx}-th subset of band_subsets is empty, cannot select a band")
    
    _, num_bands = X_2D.shape
    # Calculate/validate covariance matrix
    if covariance_matrix is None:
        covariance_matrix = np.cov(X_2D, rowvar = False)
    
    # Initialization: Randomly select one band from each subset as the initial solution
    np.random.seed(42)
    selected_indices = np.array([np.random.choice(subset, size=1)[0] for subset in band_subsets])
    iter_times = 10  # Number of iterative optimization times, consistent with the original function
    
    for iter in range(iter_times):
        # Calculate the determinant (product of singular values) of the covariance matrix for the currently selected bands
        sub_cov_matrix = covariance_matrix[np.ix_(selected_indices, selected_indices)]
        _, s, _ = np.linalg.svd(sub_cov_matrix)
        err_old = math.prod(s)
        err_new = err_old  # Record the optimal determinant value of the current iteration
        
        # Iterate over each subset position and try replacing with other bands in the subset
        for i in range(num_selected_bands):
            current_subset = band_subsets[i]  # Optional subset corresponding to the current position
            current_band = selected_indices[i]  # Band selected at the current position
            
            # Iterate over all candidate bands in the subset (excluding the currently selected one)
            for j in current_subset:
                if j == current_band:
                    continue  # Skip the currently selected band
                
                # Temporarily replace the band at the current position
                selected_indices_tmp = selected_indices.copy()
                selected_indices_tmp[i] = j
                
                # Calculate the determinant after replacement
                sub_cov_matrix_tmp = covariance_matrix[np.ix_(selected_indices_tmp, selected_indices_tmp)]
                _, s_tmp, _ = np.linalg.svd(sub_cov_matrix_tmp)
                err_tmp = math.prod(s_tmp)
                
                # If the determinant is larger after replacement, update the optimal solution
                if err_tmp > err_new:
                    err_new = err_tmp
                    selected_indices[i] = j  # Permanently replace
        
        # Convergence judgment: Terminate iteration early if the relative change is less than the threshold
        if np.abs(err_new - err_old) / np.abs(err_old + 1e-10) < 1e-6:
            break
    
    # Return the sorted selected indices (consistent with the output format of the original function)
    return np.sort(selected_indices)
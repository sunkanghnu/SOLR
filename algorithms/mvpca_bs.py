import numpy as np

def mvpca_bs(X_2D, num_selected_bands, variances = None):
    """
    Select top N features with the largest variances from 2D data matrix.
    
    Parameters:
    -----------
    X_2D : numpy.ndarray
        2D input data matrix with shape (n_samples, n_bands), where n_samples is the number of samples,
        and n_bands is the number of features/bands.
    num_selected_feature : int
        The number of top variance features to select.
    variances : numpy.ndarray, optional
        Precomputed variances of each feature/band in X_2D. If None, variances will be calculated 
        along axis 0 (across samples for each band). Default is None.
    
    Returns:
    --------
    numpy.ndarray
        Indices of the selected top variance features (sorted in ascending order of variance, 
        taking the last num_selected_feature elements which are the largest).
    """
    _, n_bands = X_2D.shape
    if (variances is None):
        variances = np.var(X_2D, axis = 0)
    # Select indices of top num_selected_feature features by variance
    selected_indices = np.argsort(variances)[-num_selected_bands:]
    return selected_indices
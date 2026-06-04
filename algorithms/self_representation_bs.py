import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed


def self_representation_bs(X_2D, num_selected_bands, correlation_matrix = None):
    """
    Perform feature (band) selection using self-representation method.
    
    Parameters:
    X_2D : ndarray, shape (n_samples, n_features)
        2D data matrix where rows represent samples and columns represent features (bands).
    num_selected_bands : int
        Number of bands (features) to select.
    correlation_matrix : ndarray, shape (n_features, n_features), optional
        Precomputed correlation matrix (X^T X) of the input data. If None, it will be computed from X_2D.
    
    Returns:
    selected_indices : ndarray, shape (num_selected_bands,)
        Indices of the selected bands (features).
    """
    _, n_bands = X_2D.shape
    if(correlation_matrix is None):
        correlation_matrix = X_2D.T @ X_2D
    
    # Initialize selected and unselected band index lists
    selected_bands = []
    unselected_bands = list(range(n_bands))
    
    # Step 1: Select the first band with the maximum variance across samples
    variances = np.var(X_2D, axis=0)
    first_band = np.argmax(variances)
    selected_bands.append(first_band)
    unselected_bands.remove(first_band)
    XTX_trace = np.trace(correlation_matrix)
    
    for selected_id in tqdm(range(1, num_selected_bands), desc='Self representation band selection', ncols=100):
        def compute_error(candidate):
            tmp_selected_bands = selected_bands.copy()
            tmp_selected_bands.append(candidate)
            XTy = correlation_matrix[:, tmp_selected_bands]
            XTyX = XTy.T @ XTy
            yTy = correlation_matrix[np.ix_(tmp_selected_bands, tmp_selected_bands)]
            yTy_inv = np.linalg.inv(yTy)
            err_tmp = XTX_trace - np.trace(yTy_inv @ XTyX)
            return candidate, err_tmp

        # Parallel compute reconstruction error for all unselected candidate bands
        results = Parallel(n_jobs=-1)(
            delayed(compute_error)(candidate) for candidate in unselected_bands
        )
        # Select the band with the minimum reconstruction error
        best_band, min_err = min(results, key=lambda x: x[1])
        selected_bands.append(best_band)
        unselected_bands.remove(best_band)
    return np.array(selected_bands)

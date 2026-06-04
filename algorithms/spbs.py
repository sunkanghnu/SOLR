import numpy as np
from typing import List, Union
from scipy.signal import convolve2d
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.feature_extraction.image import extract_patches_2d

def _compute_objective_value(A: np.ndarray, B: np.ndarray) -> float:
    """
    Compute the objective function value using optimized linear algebra operations.
    
    Implements the formula: tr(A * B^{-1} * A^T)
    Optimized using:
    1. Linear system solve instead of explicit matrix inversion
    2. Trace cyclic property: tr(A B^{-1} A^T) = tr(B^{-1} A^T A)
    
    Parameters
    ----------
    A : np.ndarray
        Matrix with shape (bands, k) where k is the number of selected bands
    B : np.ndarray
        Symmetric positive definite matrix with shape (k, k)
    reg : float, optional
        Regularization parameter to avoid singular matrices, default 1e-8
    
    Returns
    -------
    value : float
        Objective function value
    """
    # Compute A^T A (k x k matrix)
    ATA = A.T @ A
    BTB = B.T @ B
    
    
    return 2 * np.trace(ATA) - np.trace(BTB)

def _evaluate_band(band: int, selected_bands: List[int], 
                   C_X: np.ndarray) -> tuple:
    """
    Evaluate a single band candidate for parallel computation.
    
    Parameters
    ----------
    band : int
        Candidate band index
    selected_bands : List[int]
        Currently selected bands
    C_X_minus_N : np.ndarray
        Precomputed C_X - C_N matrix
    reg : float
        Regularization parameter
    
    Returns
    -------
    tuple
        (band_index, objective_value)
    """
    candidate_bands = selected_bands + [band]
    
    # Compute A and B matrices according to SOLR objective
    C_XS = C_X[:, candidate_bands]
    #A =(C_X - C_NX)[:, candidate_bands ]    
    C_S = C_X[np.ix_(candidate_bands, candidate_bands)]
    
    # Compute objective function value
    current_value = _compute_objective_value(C_XS, C_S)
    
    return band, current_value
def _evaluate_band_replace(band: int, selected_bands: List[int], repalce_index:int, 
                            C_X: np.ndarray) -> tuple:
    """
    Evaluate a single band candidate for parallel computation.
    
    Parameters
    ----------
    band : int
        Candidate band index
    selected_bands : List[int]
        Currently selected bands
    C_X_minus_N : np.ndarray
        Precomputed C_X - C_N matrix
    reg : float
        Regularization parameter
    
    Returns
    -------
    tuple
        (band_index, objective_value)
    """
    candidate_bands = selected_bands.copy()
    #print(f"original bands: {candidate_bands}")
    candidate_bands[repalce_index] = band
    #print(f"repalce bands: index= {repalce_index}, band = {band}")
    
    # Compute A and B matrices according to SOLR objective
    A = C_X[:, candidate_bands]
    #A =(C_X-C_NX)[:, candidate_bands ]
    B = C_X[np.ix_(candidate_bands, candidate_bands)]
    
    # Compute objective function value
    current_value = _compute_objective_value(A, B)
    
    return band, current_value

def sp_bs(X_2D: np.ndarray, 
            num_selected_bands: int, 
            C_X: np.ndarray = None,
            search_mode: str = 'sfs',
            n_jobs: int = -1,
            return_objective_values: bool = False) -> Union[List[int], tuple[List[int], List[float]]]:
    """
    Signal-Oriented Linear Representation Band Selection (SOLR-BS) algorithm.
    
    Selects k bands that minimize the pure signal representation error,
    derived from the core identity: ||S - S_S W||_F^2 = ||X - X_S W||_F^2 - ||N - N_S W||_F^2
    
    Parameters
    ----------
    X : np.ndarray
        Input hyperspectral data with shape (n_pixels, bands) or (height, width, bands)
    k : int
        Number of bands to select
    reg : float, optional
        Regularization parameter for numerical stability, default 1e-8
    return_objective_values : bool, optional
        If True, return the objective values of each selected band, default False
    
    Returns
    -------
    selected_bands : List[int]
        List of selected band indices in order of selection
    objective_values : List[float], optional
        List of objective values corresponding to each selected band
    """
    # Reshape 3D image to 2D pixel matrix if necessary
    n_pixels, bands = X_2D.shape
    X = X_2D

    # Convert to float64 for numerical stability
    X = X.astype(np.float64)
    
   
    # Step 2: Initialize variables
    selected_bands = []
    objective_values = []
    all_bands = set(range(bands))
    
    # Precompute X^T X once for efficiency
    if C_X is None:
        print("Precomputing X^T X...")
        X_centered = X - np.mean(X, axis = 0)
        C_X = X_centered.T @ X_centered
        
    '''
    
    low_snr_bands = set(np.where(np.diag(C_N) / np.diag(C_X) > 0.1)[0])
    if len(low_snr_bands) > bands - num_selected_bands:
        low_snr_bands= set(np.argsort(np.diag(C_X) / np.diag(C_N))[:bands - num_selected_bands - 5])
    #print(f"low_snr_bands: {low_snr_bands}")
    all_bands = all_bands.difference(low_snr_bands)
    #print(f"all_bands: {all_bands}")
    '''
    if search_mode == 'sfs':
        print(f"Starting SP-BS band selection (selecting {num_selected_bands} bands from {bands} total)...")
        first_band = np.argmax(np.diag(C_X))
        selected_bands.append(first_band)
            # Step 3: Sequential Forward Selection with parallel computation
        for i in tqdm(range(num_selected_bands - 1 ), desc="Selecting bands", ncols=100):
            remaining_bands = list(all_bands - set(selected_bands))
            
            # Parallel evaluation of all candidate bands
            results = Parallel(n_jobs = n_jobs, backend='loky')(
                delayed(_evaluate_band)(band, selected_bands, C_X)
                for band in remaining_bands
            )
            # Find the best band from parallel results
            best_band, best_value = max(results, key=lambda x: x[1])
                
            
            # Add best band to selected list
            selected_bands.append(best_band)
            objective_values.append(np.trace(C_X) - best_value)
    
    if search_mode == 'random':
        selected_bands = np.sort(np.random.choice(bands, size = num_selected_bands, replace = False))        
        iter_times = 10
        remaining_bands = list(all_bands - set(selected_bands)) 
        A = C_X[:, selected_bands]
        B = C_X[np.ix_(selected_bands, selected_bands)]          
        err_cur = _compute_objective_value(A, B)
        for iter in range(iter_times): 
            err_old = err_cur           
            for i in tqdm(range(num_selected_bands), desc = f"Selecting bands: iter #{iter+1}", ncols=100):   
                results = Parallel(n_jobs = n_jobs, backend='loky')(
                    delayed(_evaluate_band_replace)(band, selected_bands, i, C_X)
                    for band in remaining_bands
                    )             
                best_band, best_value = max(results, key=lambda x: x[1])
                if(best_value > err_old):
                    selected_bands[i] = best_band
                    err_cur = best_value
            if(np.abs(err_cur - err_old)/np.abs(err_old + 1e-10) < 1e-6):
                break  
            objective_values.append(err_cur)
    
    if return_objective_values:
        return np.array(selected_bands), objective_values
    else:
        return np.array(selected_bands)
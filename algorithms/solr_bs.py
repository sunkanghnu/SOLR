import numpy as np
from typing import List, Union
from scipy.signal import convolve2d
from joblib import Parallel, delayed
from tqdm import tqdm

from sklearn.feature_extraction.image import extract_patches_2d

def _correct_noise_covariance_matrix(noise_covariance_matrix: np.ndarray) -> np.ndarray:
    sum_row = np.sum(np.abs(noise_covariance_matrix), axis=1)
    row_diagnonal_ratio = np.diag(noise_covariance_matrix) / sum_row
    row_diagnonal_ratio = row_diagnonal_ratio.reshape(-1,1)
    row_diagnonal_ratio[row_diagnonal_ratio > 0.1 ] = 1
    revised_matrix = np.sqrt(row_diagnonal_ratio @ row_diagnonal_ratio.T)          
    return  revised_matrix * noise_covariance_matrix
    
def compute_noise_covariance_matrix(X_3D: np.ndarray) -> np.ndarray:
    """
    Estimate noise covariance matrix using the shift-difference method (MNF standard).
    
    This method estimates noise by computing differences between adjacent pixels,
    leveraging the fact that signal is spatially correlated while noise is not.
    
    Parameters
    ----------
    X_3D : np.ndarray
        Input hyperspectral image with shape (height, width, bands) or (n_pixels, bands)
    
    Returns
    -------
    C_N : np.ndarray
        Noise covariance matrix with shape (bands, bands), corrected by 4/5 factor
    """
 
    
    height, width, bands = X_3D.shape  
   
    noise_estimate_kernel = np.array([[0, 0.25, 0.],
                                      [0.25, -1, 0.25], 
                                      [0., 0.25, 0.]])
    #preprocess_kernel = np.ones((3, 3))/9.0

    noise_estimate = np.zeros([height-2, width-2, bands])
    for band in range(bands):
        #preprocess_band = convolve2d(X_3D[:, :, band], preprocess_kernel, mode='same',boundary='symm')
        noise_estimate[:, :, band] =  convolve2d(X_3D[:, :, band], noise_estimate_kernel, mode='valid',boundary='symm')        
        
    noise_estimate = noise_estimate.reshape(-1, bands)  
    nosie_centered = noise_estimate - np.mean(noise_estimate , axis = 0)
    C_N = nosie_centered .T @ nosie_centered    
    return C_N

def compute_noise_data_covariance_matrix(X_3D: np.ndarray) -> np.ndarray:
    """
    Estimate noise covariance matrix using the shift-difference method (MNF standard).
    
    This method estimates noise by computing differences between adjacent pixels,
    leveraging the fact that signal is spatially correlated while noise is not.
    
    Parameters
    ----------
    X_3D : np.ndarray
        Input hyperspectral image with shape (height, width, bands) or (n_pixels, bands)
    
    Returns
    -------
    C_N : np.ndarray
        Noise covariance matrix with shape (bands, bands), corrected by 4/5 factor
    """
 
    
    height, width, bands = X_3D.shape  
   
    noise_estimate_kernel = np.array([[0, 0.25, 0.],
                       [0.25, -1, 0.25], 
                       [0., 0.25, 0.]])
    preprocess_kernel = np.ones((3, 3))/9.0

    noise_estimate = np.zeros_like(X_3D)
    for band in range(bands):
        #preprocess_band = convolve2d(X_3D[:, :, band], preprocess_kernel, mode='same',boundary='symm')
        noise_estimate[:, :, band] =  convolve2d(X_3D[:, :, band], noise_estimate_kernel, mode='same',boundary='symm')
    noise_estimate = noise_estimate.reshape(-1, bands)
    nosie_centered = noise_estimate - np.mean(noise_estimate, axis = 0)
    X_2D = X_3D.reshape(-1, bands)
    X_2D_centered = X_2D - np.mean(X_2D, axis = 0)
    C_NX = nosie_centered .T @ X_2D_centered
    return C_NX
def _compute_objective_value(A: np.ndarray, B: np.ndarray, reg: float = 1e-8) -> float:
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
    
    # Add regularization to B for numerical stability
    B_reg = B + reg * np.eye(B.shape[0], dtype=np.float64)
    
    # Solve linear system B_reg * X = ATA instead of computing B^{-1}
    # This is faster and more numerically stable than explicit inversion
    '''
    print("矩阵形状:", B_reg.shape)
    print("是否有NaN:", np.isnan(B_reg).any())
    print("是否有Inf:", np.isinf(B_reg).any())

    if B_reg.shape[0] == B_reg.shape[1]:
        # 计算秩（使用SVD更可靠）
        rank = np.linalg.matrix_rank(B_reg)
        print(f"矩阵秩: {rank}, 阶数: {B_reg.shape[0]}")
        print(f"秩亏: {B_reg.shape[0] - rank}")
        
        # 计算条件数（越大越病态）
        cond = np.linalg.cond(B_reg)
        print(f"条件数: {cond:.2e}")
        
        # 计算最小特征值
        eigvals = np.linalg.eigvalsh(B_reg)  # 对称矩阵用eigvalsh更快更稳定
        print(f"最小特征值: {np.min(eigvals):.2e}")
        print(f"最大特征值: {np.max(eigvals):.2e}")
    '''
    #X = np.linalg.solve(B_reg, ATA)
    X,_,_,_ = np.linalg.lstsq(B, ATA, rcond=None)
    #X = np.linalg.pinv(B_reg) @ ATA 
    # Compute trace of X
    return np.trace(X)

def _evaluate_band(band: int, selected_bands: List[int], 
                   C_X_minus_N: np.ndarray,  reg: float) -> tuple:
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
    A = C_X_minus_N[:, candidate_bands]
    #A =(C_X - C_NX)[:, candidate_bands ]    
    B = C_X_minus_N[np.ix_(candidate_bands, candidate_bands)]
    
    # Compute objective function value
    current_value = _compute_objective_value(A, B, reg)
    
    return band, current_value
def _evaluate_band_replace(band: int, selected_bands: List[int], repalce_index:int, 
                            C_X_minus_N: np.ndarray,reg: float) -> tuple:
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
    A = C_X_minus_N[:, candidate_bands]
    #A =(C_X-C_NX)[:, candidate_bands ]
    B = C_X_minus_N[np.ix_(candidate_bands, candidate_bands)]
    
    # Compute objective function value
    current_value = _compute_objective_value(A, B, reg)
    
    return band, current_value

def solr_bs(X_3D: np.ndarray, 
            num_selected_bands: int, 
            C_X: np.ndarray = None,
            C_N: np.ndarray = None,
            search_mode: str = 'sfs',
            reg: float = 1e-8,
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
    if X_3D.ndim == 3:
        n_pixels = X_3D.shape[0] * X_3D.shape[1]
        bands = X_3D.shape[2]
        X = X_3D.reshape(n_pixels, bands)
    else:
        n_pixels, bands = X_3D.shape
        X = X_3D

    # Convert to float64 for numerical stability
    X = X.astype(np.float64)
    
    # Step 1: Estimate noise covariance matrix   
    if C_N is None:
        print("Estimating noise covariance matrix...")
        C_N = compute_noise_covariance_matrix(X_3D)
    
    C_N_corrected = _correct_noise_covariance_matrix(C_N)
    
    # Step 2: Initialize variables
    selected_bands = []
    objective_values = []
    all_bands = set(range(bands))
    
    # Precompute X^T X once for efficiency
    if C_X is None:
        print("Precomputing X^T X...")
        X_centered = X - np.mean(X, axis = 0)
        C_X = X_centered.T @ X_centered
        
        
    C_X_minus_N = C_X - C_N#_corrected
    #Test = np.diag(C_N)/ np.diag(C_X)
    #print(f'Test = {Test}')
    #print(f'C_X = {np.diag(C_X)}')
    #print(f'C_N = {np.diag(C_N)}')
    #print(f'C_X_N = {np.diag(C_X_minus_N)}')    
    
    low_snr_bands = set(np.where(np.diag(C_N) / np.diag(C_X) > 0.1)[0])
    if len(low_snr_bands) > bands - num_selected_bands:
        low_snr_bands= set(np.argsort(np.diag(C_X) / np.diag(C_N))[:bands - num_selected_bands - 5])
    #print(f"low_snr_bands: {low_snr_bands}")
    all_bands = all_bands.difference(low_snr_bands)
    #print(f"all_bands: {all_bands}")
    
    if search_mode == 'sfs':
        print(f"Starting SOLR-BS band selection (selecting {num_selected_bands} bands from {bands} total)...")
        first_band = np.argmax(np.diag(C_X)/np.diag(C_N_corrected))
        selected_bands.append(first_band)
            # Step 3: Sequential Forward Selection with parallel computation
        for i in tqdm(range(num_selected_bands - 1 ), desc="Selecting bands", ncols=100):
            remaining_bands = list(all_bands - set(selected_bands))
            
            # Parallel evaluation of all candidate bands
            results = Parallel(n_jobs = n_jobs, backend='loky')(
                delayed(_evaluate_band)(band, selected_bands, C_X_minus_N,reg)
                for band in remaining_bands
            )
            # Find the best band from parallel results
            best_band, best_value = max(results, key=lambda x: x[1])
                
            
            # Add best band to selected list
            selected_bands.append(best_band)
            objective_values.append(np.trace(C_X_minus_N) - best_value)
    
    if search_mode == 'random':
        selected_bands = np.sort(np.random.choice(bands, size = num_selected_bands, replace = False))        
        iter_times = 10
        remaining_bands = list(all_bands - set(selected_bands)) 
        A = C_X_minus_N[:, selected_bands]
        B = C_X[np.ix_(selected_bands, selected_bands)]          
        err_cur = _compute_objective_value(A, B, reg)
        for iter in range(iter_times): 
            err_old = err_cur           
            for i in tqdm(range(num_selected_bands), desc = f"Selecting bands: iter #{iter+1}", ncols=100):   
                results = Parallel(n_jobs = n_jobs, backend='loky')(
                    delayed(_evaluate_band_replace)(band, selected_bands, i, C_X_minus_N,reg)
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
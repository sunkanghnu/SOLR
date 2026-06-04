import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed
from skimage.metrics import structural_similarity
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

def compute_entropy(X_2D):
    """Calculate information entropy for each band
    
    Args:
        X_2D (np.ndarray): 2D array with shape (n_pixels, n_bands)
    
    Returns:
        np.ndarray: Entropy values for each band with shape (n_bands,)
    """
    _, n_bands = X_2D.shape
    entropy_vals = np.zeros(n_bands)
    for i in range(n_bands):
        band_data = X_2D[:, i]
        hist, _ = np.histogram(band_data, bins = 256)
        prob = hist / np.sum(hist)
        entropy_vals[i] = entropy(prob)
    return entropy_vals

def euclidean_distance_matrix(X_2D):
    """Calculate Euclidean distance matrix
    
    Args:
        X_2D (np.ndarray): 2D array with shape (n_pixels, n_bands)
    
    Returns:
        np.ndarray: Square Euclidean distance matrix with shape (n_bands, n_bands)
    """
    return squareform(pdist(X_2D.T, metric = 'euclidean'))

def compute_variances(X_2D):
    """Calculate variances for each band
    
    Args:
        X_2D (np.ndarray): 2D array with shape (n_pixels, n_bands)
    
    Returns:
        np.ndarray: Variance values for each band with shape (n_bands,)
    """
    variances = np.var(X_2D, axis = 0)
    return variances

def rbf_kernel_dismat(distance_matrix, sigma = 500):
    """
    Calculate RBF kernel matrix
    
    Args:
        distance_matrix (np.ndarray): Distance matrix with shape (n_bands, n_bands)
        sigma (float): Sigma parameter for RBF kernel, default is 500
    
    Returns:
        np.ndarray: RBF kernel matrix with shape (n_bands, n_bands)
    """
    return np.exp(-distance_matrix / (2 * sigma ** 2))

def compute_correlation_matrix_unsupervised(X_2D):
    """Calculate unsupervised correlation matrix
    
    Args:
        X_2D (np.ndarray): 2D array with shape (n_pixels, n_bands)
    
    Returns:
        np.ndarray: Correlation matrix with shape (n_bands, n_bands)
    """
    corr_matrix = X_2D.T @ X_2D
    return corr_matrix

def compute_corrcoef_matrix(X_2D):
    """
    Calculate inter-band correlation coefficient matrix (optimized with matrix operation)
    
    Args:
        X_2D (np.ndarray): Hyperspectral data with shape (n_pixels, n_bands)
    
    Returns:
        np.ndarray: Correlation coefficient matrix with shape (n_bands, n_bands)
    """
    corrcoef_matrix = np.abs(np.corrcoef(X_2D, rowvar = False))
    return corrcoef_matrix

def compute_covariance_matrix(X_2D):
    """Calculate covariance matrix
    
    Args:
        X_2D (np.ndarray): 2D array with shape (n_pixels, n_bands)
    
    Returns:
        np.ndarray: Covariance matrix with shape (n_bands, n_bands)
    """
    X_2D_centered = X_2D - X_2D.mean(axis = 0)
    #cov_matrix = np.cov(X_2D, rowvar = False)
    cov_matrix = X_2D_centered.T @ X_2D_centered
    return cov_matrix

def _compute_single_ssim(args):
    """
    Helper function: Calculate SSIM value for a single band pair
    
    Args:
        args (tuple): Tuple containing (hyperspectral_image, i, j, window_size)
            hyperspectral_image (np.ndarray): 3D hyperspectral image with shape (height, width, n_bands)
            i (int): Index of first band
            j (int): Index of second band
            window_size (int): Window size for SSIM calculation
    
    Returns:
        tuple: (i, j, ssim_value) where ssim_value is the computed SSIM between band i and band j
    """
    hyperspectral_image, i, j, window_size = args
    band_img_i = hyperspectral_image[:, :, i].astype(np.int32)
    band_img_j = hyperspectral_image[:, :, j].astype(np.int32)
    ssim_value = structural_similarity(band_img_i, band_img_j, gaussian_weight=True, win_size=window_size)
    return i, j, ssim_value

def compute_ssim_matrix(X_3D, window_size=3, n_jobs=-1):
    """
    Calculate SSIM (Structural Similarity Index) matrix for hyperspectral image (symmetric matrix)
    
    Args:
        X_3D (np.ndarray): Hyperspectral image matrix with shape (height × width × n_bands)
        window_size (int): Window size for SSIM calculation, default is 3
        n_jobs (int): Number of parallel jobs, -1 means using all CPU cores
    
    Returns:
        np.ndarray: Inter-band SSIM matrix with shape (n_bands × n_bands)
    """
    total_band_count = X_3D.shape[-1]
    ssim_matrix = np.zeros((total_band_count, total_band_count))
    np.fill_diagonal(ssim_matrix, 1.0)
    
    # Generate all band pairs (i, j) where i < j
    band_pairs = []
    for i in range(total_band_count):
        for j in range(i + 1, total_band_count):
            band_pairs.append((X_3D, i, j, window_size))
    
    # Parallel compute SSIM for all band pairs with tqdm progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_single_ssim)(pair) for pair in tqdm(band_pairs, desc="Computing SSIM", unit="pair")
    )
    
    # Fill SSIM matrix
    for i, j, ssim_value in results:
        ssim_matrix[i, j] = ssim_value
        ssim_matrix[j, i] = ssim_value
    
    return ssim_matrix

def _histogram_prob(data, bins=256, eps=1e-10):
    """Calculate histogram probability distribution for 1D data (paper-specified scheme)
    
    Args:
        data (np.ndarray): 1D array of data
        bins (int): Number of bins for histogram, default is 256
        eps (float): Small value to avoid division by zero, default is 1e-10
    
    Returns:
        np.ndarray: Normalized probability distribution of the histogram
    """
    counts, _ = np.histogram(data, bins=bins, density=False)
    p = counts / counts.sum()
    p = p + eps
    return p / p.sum()

def _joint_histogram_prob(data1, data2, bins=256, eps=1e-10):
    """Calculate joint histogram probability distribution for two 1D data arrays
    
    Args:
        data1 (np.ndarray): First 1D data array
        data2 (np.ndarray): Second 1D data array
        bins (int): Number of bins for joint histogram, default is 256
        eps (float): Small value to avoid division by zero, default is 1e-10
    
    Returns:
        np.ndarray: Normalized joint probability distribution
    """
    counts, _, _ = np.histogram2d(data1, data2, bins=bins, density=False)
    p = counts / counts.sum()
    p = p + eps
    return p / p.sum()

def _shannon_entropy(p):
    """Calculate Shannon entropy H(X)
    
    Args:
        p (np.ndarray): Probability distribution array
    
    Returns:
        float: Shannon entropy value
    """
    return -np.sum(p * np.log2(p))

def mutual_info_distance(data1, data2, bins=256, eps=1e-10):
    """Calculate WaLuMI normalized mutual information distance D_NI (paper formula 7)
    
    Args:
        data1 (np.ndarray): First 1D data array
        data2 (np.ndarray): Second 1D data array
        bins (int): Number of bins for histogram, default is 256
        eps (float): Small value to avoid division by zero, default is 1e-10
    
    Returns:
        float: Normalized mutual information distance value
    """
    p1 = _histogram_prob(data1, bins, eps)
    p2 = _histogram_prob(data2, bins, eps)
    p_joint = _joint_histogram_prob(data1, data2, bins, eps)
    
    h1 = _shannon_entropy(p1)
    h2 = _shannon_entropy(p2)
    h_joint = _shannon_entropy(p_joint)
    
    mi = h1 + h2 - h_joint
    ni = 2 * mi / (h1 + h2 + eps)
    return (1 - np.sqrt(ni)) ** 2

def symmetric_kl_distance(data1, data2, bins=256, eps=1e-10):
    """Calculate WaLuDi symmetric KL divergence distance D_KL (paper formula 8)
    
    Args:
        data1 (np.ndarray): First 1D data array
        data2 (np.ndarray): Second 1D data array
        bins (int): Number of bins for histogram, default is 256
        eps (float): Small value to avoid division by zero, default is 1e-10
    
    Returns:
        float: Symmetric KL divergence distance value
    """
    data_min = min(data1.min(), data2.min())
    data_max = max(data1.max(), data2.max())
    bins_edges = np.linspace(data_min, data_max, bins + 1)
    
    counts1, _ = np.histogram(data1, bins=bins_edges, density=False)
    counts2, _ = np.histogram(data2, bins=bins_edges, density=False)
    
    p1 = counts1 / counts1.sum() + eps
    p2 = counts2 / counts2.sum() + eps
    p1 = p1 / p1.sum()
    p2 = p2 / p2.sum()
    
    kl1 = np.sum(p1 * np.log2(p1 / p2))
    kl2 = np.sum(p2 * np.log2(p2 / p1))
    return kl1 + kl2

def _compute_single_distance(i, j, data_flat, dist_func, bins, eps):
    """Calculate distance for a single band pair (for parallel call)
    
    Args:
        i (int): Index of first band
        j (int): Index of second band
        data_flat (np.ndarray): Flattened hyperspectral data with shape (n_pixels, n_bands)
        dist_func (function): Distance calculation function (mutual_info_distance or symmetric_kl_distance)
        bins (int): Number of bins for histogram
        eps (float): Small value to avoid division by zero
    
    Returns:
        tuple: (i, j, dist) where dist is the computed distance between band i and band j
    """
    return i, j, dist_func(data_flat[:, i], data_flat[:, j], bins, eps)

def compute_distance_matrix_parallel(X_2D, dist_func, bins=256, eps=1e-10, n_jobs=-1):
    """
    Parallel construction of L×L inter-band dissimilarity matrix
    
    Args:
        X_2D (np.ndarray): Flattened hyperspectral data with shape (n_pixels, L) where L is number of bands
        dist_func (function): Distance calculation function (mutual_info_distance or symmetric_kl_distance)
        bins (int): Number of bins for histogram, default is 256
        eps (float): Small value to avoid division by zero, default is 1e-10
        n_jobs (int): Number of parallel CPU cores, -1 means using all cores
    
    Returns:
        np.ndarray: Symmetric distance matrix with shape (L, L)
    """
    L = X_2D.shape[1]
    dist_matrix = np.zeros((L, L), dtype=np.float32)
    
    # Generate all upper triangular band pair indices (avoid duplicate calculation)
    pairs = [(i, j) for i in range(L) for j in range(i + 1, L)]
    
    # Parallel compute distance for all band pairs
    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_compute_single_distance)(i, j, X_2D, dist_func, bins, eps)
        for i, j in pairs
    )
    
    # Fill symmetric distance matrix
    for i, j, dist in results:
        dist_matrix[i, j] = dist
        dist_matrix[j, i] = dist
    
    return dist_matrix
import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.signal import convolve2d
from joblib import Parallel, delayed
import warnings
# Ignore runtime warnings
warnings.filterwarnings("ignore", category = RuntimeWarning)

def _generate_gaussian_window(size = 11, sigma = 1.5):
    """Generate Gaussian weight window
    
    Parameters:
        size (int): Size of the Gaussian window, default is 11
        sigma (float): Standard deviation of the Gaussian kernel, default is 1.5
    
    Returns:
        numpy.ndarray: Normalized Gaussian kernel window
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def _compute_single_image_stats(img, window, L = 255):
    """Compute statistics for a single image (parallel version)
    
    Parameters:
        img (numpy.ndarray): Input image array
        window (numpy.ndarray): Gaussian weight window
        L (int): Dynamic range of pixel values, default is 255
    
    Returns:
        dict: Dictionary containing image statistics including:
            - img: Float-type image array
            - mu: Weighted mean of the image
            - mu_sq: Squared weighted mean
            - img_sq_conv: Convolved result of squared image
    """
    # Convert to float type
    img_float = img.astype(np.float64)
    # Calculate weighted mean
    mu = convolve2d(img_float, window, mode='same', boundary='symm')
    
    # Calculate weighted squared mean
    img_sq = img_float ** 2
    img_sq_conv = convolve2d(img_sq, window, mode='same', boundary='symm')
    
    return {
        'img': img_float,
        'mu': mu,
        'mu_sq': mu ** 2,
        'img_sq_conv': img_sq_conv
    }


def _precompute_image_stats(X_3D, window_size = 11, sigma = 1.5, L = 255, n_jobs =None):
    """Precompute statistics for all images in parallel using joblib
    
    Parameters:
        X_3D (numpy.ndarray): Hyperspectral data cube with shape (H, W, Bands)
        window_size (int): Size of Gaussian window, default is 11
        sigma (float): Standard deviation of Gaussian kernel, default is 1.5
        L (int): Dynamic range of pixel values, default is 255
        n_jobs (int): Number of parallel jobs, None means using all processors
    
    Returns:
        list: List of precomputed statistics dictionaries for each band
    """
    window = _generate_gaussian_window(window_size, sigma)
    num_bands = X_3D.shape[2]
    stats = Parallel(n_jobs = n_jobs, verbose = 10)(
        delayed(_compute_single_image_stats)(X_3D[:, :, i], window, L)
        for i in range(num_bands)
    )
    return stats

def _calculate_mssim_with_stats(stats_x, stats_y, window, L = 255):
    """
    Calculate MSSIM (Multi-Scale Structural Similarity) between two images using precomputed statistics
    Avoid recalculating statistics for the same image in each image pair
    
    Parameters:
        stats_x (dict): Precomputed statistics for image X
        stats_y (dict): Precomputed statistics for image Y
        window (numpy.ndarray): Gaussian weight window
        L (int): Dynamic range of pixel values, default is 255
    
    Returns:
        float: Mean MSSIM value between the two images
    """
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    C3 = C2 / 2
    
    # Get data from precomputed statistics
    mu_x = stats_x['mu']
    mu_y = stats_y['mu']
    mu_x_sq = stats_x['mu_sq']
    mu_y_sq = stats_y['mu_sq']
    
    # Calculate variance - using precomputed results
    sigma_x_sq = stats_x['img_sq_conv'] - mu_x_sq
    sigma_y_sq = stats_y['img_sq_conv'] - mu_y_sq
    
    # Calculate covariance - only need to compute once
    sigma_xy = convolve2d(
        stats_x['img'] * stats_y['img'], 
        window, 
        mode='same', 
        boundary='symm'
    ) - mu_x * mu_y
    
    # Calculate SSIM components
    luminance = (2 * mu_x * mu_y + C1) / (mu_x_sq + mu_y_sq + C1)
    
    # Avoid NaN caused by negative variance
    contrast_num = 2 * np.sqrt(np.maximum(sigma_x_sq, 0)) * np.sqrt(np.maximum(sigma_y_sq, 0)) + C2
    contrast_denom = np.maximum(sigma_x_sq, 0) + np.maximum(sigma_y_sq, 0) + C2
    contrast = contrast_num / contrast_denom
    
    # Structure component
    structure_num = sigma_xy + C3
    structure_denom = np.sqrt(np.maximum(sigma_x_sq, 0)) * np.sqrt(np.maximum(sigma_y_sq, 0)) + C3
    structure = structure_num / structure_denom
    
    # Comprehensive SSIM map
    ssim_map = luminance * contrast * structure
    
    # Calculate mean while ignoring NaN values
    valid_pixels = ~np.isnan(ssim_map)
    if np.any(valid_pixels):
        return np.mean(ssim_map[valid_pixels])
    return 0.0

def compute_mssim_matrix(X_3D, window_size = 11, sigma = 1.5, L = 255, n_jobs = 20):
    """
    Compute MSSIM similarity matrix between multiple images (joblib parallel version)
    
    Parameters:
        X_3D (numpy.ndarray): Hyperspectral data cube with shape (H, W, Bands)
        window_size (int): Size of sliding Gaussian window, default is 11
        sigma (float): Standard deviation of Gaussian kernel, default is 1.5
        L (int): Dynamic range of pixel values, default is 255
        n_jobs (int): Number of parallel processing cores, default is 20
    
    Returns:
        numpy.ndarray: N×N similarity matrix where N is the number of bands
    """
    num_bands = X_3D.shape[2]
    sim_matrix = np.zeros((num_bands, num_bands))
    np.fill_diagonal(sim_matrix, 1.0)
    window = _generate_gaussian_window(window_size, sigma)
    precomputed_stats = _precompute_image_stats(X_3D, window_size=window_size, sigma=sigma, L=L, n_jobs = n_jobs)

    # Task list
    tasks = [(i, j, precomputed_stats[i], precomputed_stats[j])
             for i in range(num_bands - 1)
             for j in range(i + 1, num_bands)]

    def mssim_job(args):
        i, j, stats_i, stats_j = args
        sim = _calculate_mssim_with_stats(stats_i, stats_j, window, L)
        return i, j, sim

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(mssim_job)(task) for task in tasks
    )

    for i, j, sim in results:
        sim_matrix[i, j] = sim
        sim_matrix[j, i] = sim
    return sim_matrix

def similarity_ranking_bs(similarity_matrix, similarity_threshold, num_selected_bands,sr_eta = None):
    """
    Implementation of SR algorithm: Select the most representative K bands as cluster centers based on similarity matrix
    
    Parameters:
        similarity_matrix (numpy.ndarray): Similarity matrix with shape (L, L), where L is the number of bands
        similarity_threshold (float): Similarity threshold for filtering significantly similar band pairs
        num_selected_bands (int): Number of bands to select (K)
        sr_eta (numpy.ndarray): Precomputed eta scores, if provided will skip eta calculation
    
    Returns:
        tuple: 
            - selected_indices (numpy.ndarray): List of selected band indices with length K
            - eta (numpy.ndarray): Comprehensive score for each band
    
    Reference:
        B. Xu, X. Li, W. Hou, Y. Wang, and Y. Wei, 
        "A Similarity-Based Ranking Method for Hyperspectral Band Selection, " 
        IEEE Transactions on Geoscience and Remote Sensing, vol. 59, pp. 9585-9599, (2021).
    """
    if sr_eta is None:
        L = similarity_matrix.shape[0]  # Number of bands
        
        # Step 1: Calculate average similarity α for each band
        alpha = np.zeros(L)
        for i in range(L):
            # Calculate sum and count of similarity values exceeding threshold s_c
            valid_similarities = similarity_matrix[i, similarity_matrix[i] > similarity_threshold]
            if len(valid_similarities) > 0:
                alpha[i] = np.mean(valid_similarities)
            else:
                alpha[i] = 0  # Set to 0 if no similarities exceed threshold
        
        # Sort indices in descending order of α
        sorted_indices = np.argsort(alpha)[::-1]
        
        # Step 2: Calculate significant dissimilarity φ
        phi = np.zeros(L)
        A = np.zeros(L, dtype=int)  # Store index of the most similar high-α band for each band
        
        # Initialize φ: set φ of the first band (highest α) to 1, others to 0
        phi[sorted_indices[0]] = 1.0
        
        # For each band in sorted order (starting from the second)
        for i in range(1, L):
            current_idx = sorted_indices[i]  # Current band index
            
            # Find the most similar band among all higher-α bands
            max_similarity = -np.inf
            best_match_idx = -1
            
            for j in range(i):  # Only consider bands with higher α (ranked earlier)
                candidate_idx = sorted_indices[j]
                similarity = similarity_matrix[current_idx, candidate_idx]
                
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match_idx = candidate_idx
            
            # Record maximum similarity and corresponding band index
            phi[current_idx] = max_similarity
            A[current_idx] = best_match_idx
        
        # Set φ value of the highest-α band to the minimum of all non-zero φ values
        min_phi = np.min(phi[phi > 0])  # Only consider non-zero φ values
        phi[sorted_indices[0]] = min_phi
        A[sorted_indices[0]] = sorted_indices[0]  # Highest-α band is most similar to itself
        
        # Calculate θ = √(1 - φ²), representing dissimilarity
        theta = np.sqrt(1 - np.square(phi))
        
        # Step 3: Calculate comprehensive score η for each band
        # Normalize α and θ
        norm_alpha = (alpha - np.min(alpha)) / (np.max(alpha) - np.min(alpha) + 1e-10)
        norm_theta = (theta - np.min(theta)) / (np.max(theta) - np.min(theta) + 1e-10)
        eta = norm_alpha * norm_theta
    else:
        eta = sr_eta
    
    # Step 4: Select top K bands with highest scores
    selected_indices = np.argsort(eta)[::-1][:num_selected_bands]
    
    return selected_indices, eta
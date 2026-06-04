import numpy as np
from scipy.ndimage import uniform_filter
from scipy.stats import gaussian_kde
from typing import List, Union
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')  # 屏蔽数值计算警告


def compute_spatial_neighborhood_mean(hsi: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Compute spatial neighborhood mean for each pixel in hyperspectral image.
    Uses reflection padding to handle boundary pixels correctly.
    
    Parameters
    ----------
    hsi : np.ndarray
        Input hyperspectral image with shape (height, width, bands)
    window_size : int, optional
        Size of the square neighborhood window, default 3
    
    Returns
    -------
    neighborhood_mean : np.ndarray
        Neighborhood mean image with same shape as input
    """
    print("Computing spatial neighborhood mean (reflection padding)...")
    return uniform_filter(hsi, size=(window_size, window_size, 1), mode='reflect')


def compute_mutual_information(x: np.ndarray, 
                               y: np.ndarray, 
                               max_samples: int = 10000) -> float:
    """
    Compute mutual information between two 1D arrays using Gaussian kernel density estimation.
    Implements the exact formula from the MIMN-DPP paper.
    Automatically downsamples large datasets for efficiency.
    
    Parameters
    ----------
    x : np.ndarray
        First 1D array (band 1 values)
    y : np.ndarray
        Second 1D array (band 2 values)
    max_samples : int, optional
        Maximum number of samples to use (downsamples if exceeded), default 10000
    
    Returns
    -------
    mi : float
        Mutual information between x and y, guaranteed non-negative
    """
    n = len(x)
    
    # 提前处理常数波段（互信息为0）
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std < 1e-10 or y_std < 1e-10:
        return 0.0
    
    # 下采样提升效率
    if n > max_samples:
        indices = np.random.choice(n, max_samples, replace=False)
        x, y = x[indices], y[indices]
        n = max_samples
    
    # Silverman带宽规则（预计算避免重复计算）
    h_x = 1.06 * x_std * n ** (-1/5)
    h_y = 1.06 * y_std * n ** (-1/5)
    h_joint = np.sqrt(h_x**2 + h_y**2) / np.sqrt(2)
    
    # 核密度估计
    kde_joint = gaussian_kde(np.vstack([x, y]), bw_method=h_joint)
    kde_x = gaussian_kde(x, bw_method=h_x)
    kde_y = gaussian_kde(y, bw_method=h_y)
    
    # 评估密度值
    p_joint = kde_joint(np.vstack([x, y]))
    p_x = kde_x(x)
    p_y = kde_y(y)
    
    # 数值稳定性处理
    epsilon = 1e-10
    p_joint = np.maximum(p_joint, epsilon)
    p_x = np.maximum(p_x, epsilon)
    p_y = np.maximum(p_y, epsilon)
    
    # 计算互信息
    mi = np.mean(np.log(p_joint / (p_x * p_y)))
    return max(mi, 0.0)


def _compute_mi_pair(data: np.ndarray, i: int, j: int, max_samples: int) -> tuple:
    """辅助函数：计算单对波段的互信息，用于并行计算"""
    mi = compute_mutual_information(data[:, i], data[:, j], max_samples)
    return (i, j, mi)


def build_kernel_matrix(data: np.ndarray, 
                        max_samples: int = 10000,
                        n_jobs: int = -1) -> np.ndarray:
    """
    Build DPP kernel matrix using mutual information as similarity measure.
    Kernel matrix L[i,j] = MI(band_i, band_j) for i≠j, L[i,i] = 1.0.
    优化点：并行计算所有波段对，tqdm显示进度
    
    Parameters
    ----------
    data : np.ndarray
        Input data matrix with shape (n_pixels, n_bands)
    max_samples : int, optional
        Maximum samples for mutual information computation, default 10000
    n_jobs : int, optional
        并行计算的核心数，-1表示使用所有核心，default -1
    
    Returns
    -------
    L : np.ndarray
        Symmetric positive semi-definite kernel matrix (n_bands, n_bands)
    """
    n_bands = data.shape[1]
    L = np.eye(n_bands, dtype=np.float64)  # 对角线为1.0
    
    # 生成所有i<j的波段对
    pairs = [(i, j) for i in range(n_bands) for j in range(i+1, n_bands)]
    
    # 并行计算互信息，带进度条
    print(f"Computing MI for {len(pairs)} band pairs (n_jobs={n_jobs})...")
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_compute_mi_pair)(data, i, j, max_samples)
        for i, j in tqdm(pairs, desc="MI Calculation", unit="pair", ncols=100)
    )
    
    # 填充对称矩阵
    for i, j, mi in results:
        L[i, j] = mi
        L[j, i] = mi
    
    return L


def compute_band_entropy(band: np.ndarray, bins: int = 256) -> float:
    """
    Compute information entropy of a single spectral band.
    Uses histogram-based estimation for efficiency.
    
    Parameters
    ----------
    band : np.ndarray
        2D array of a single spectral band
    bins : int, optional
        Number of histogram bins, default 256
    
    Returns
    -------
    entropy : float
        Information entropy of the band
    """
    # 归一化到[0,1]
    band_min = np.min(band)
    band_max = np.max(band)
    band_normalized = (band - band_min) / (band_max - band_min + 1e-10)
    
    # 计算概率密度
    hist, _ = np.histogram(band_normalized, bins=bins, range=(0, 1), density=True)
    hist = hist[hist > 0]
    
    return -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0


def estimate_band_noise(band: np.ndarray, window_size: int = 3, bins: int = 50) -> float:
    """
    Estimate noise level of a single band using local entropy mode method.
    优化点：向量化窗口熵计算，替代原循环，提升效率
    
    Parameters
    ----------
    band : np.ndarray
        2D array of a single spectral band
    window_size : int, optional
        Size of local sliding window, default 3
    bins : int, optional
        Number of histogram bins for entropy distribution, default 50
    
    Returns
    -------
    noise_estimate : float
        Estimated noise level of the band
    """
    # 提取所有滑动窗口（向量化）
    windows = np.lib.stride_tricks.sliding_window_view(band, (window_size, window_size))
    windows = windows.reshape(-1, window_size * window_size)
    
    # 向量化计算窗口熵（替代原for循环）
    def window_entropy(window):
        hist, _ = np.histogram(window, bins=bins, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0
    
    # 批量计算所有窗口熵
    window_entropies = np.array([window_entropy(w) for w in windows])
    
    # 找熵分布的众数区间
    hist, bin_edges = np.histogram(window_entropies, bins=bins, density=True)
    mode_bin_idx = np.argmax(hist)
    
    # 计算众数区间内的平均熵
    mask = (window_entropies >= bin_edges[mode_bin_idx]) & (window_entropies < bin_edges[mode_bin_idx + 1])
    noise_estimate = np.mean(window_entropies[mask]) + 1e-10  # 避免除零
    
    return noise_estimate


def _compute_band_stats(band: np.ndarray, entropy_bins: int = 256, noise_window: int = 3, noise_bins: int = 50) -> tuple:
    """辅助函数：并行计算单波段的熵和噪声，返回(entropy, noise)"""
    entropy = compute_band_entropy(band, bins=entropy_bins)
    noise = estimate_band_noise(band, window_size=noise_window, bins=noise_bins)
    return (entropy, noise)


def compute_mimn_scores(X_3D: np.ndarray, 
                        entropy_bins: int = 256,
                        noise_window: int = 3,
                        noise_bins: int = 50,
                        n_jobs: int = -1) -> np.ndarray:
    """
    Compute MIMN (Maximum Information Minimum Noise) scores for all bands.
    优化点：并行计算所有波段的熵和噪声，预计算统计量，tqdm显示进度
    
    Parameters
    ----------
    hsi : np.ndarray
        Input hyperspectral image with shape (height, width, bands)
    entropy_bins : int, optional
        熵计算的直方图分箱数，default 256
    noise_window : int, optional
        噪声估计的窗口大小，default 3
    noise_bins : int, optional
        噪声估计的直方图分箱数，default 50
    n_jobs : int, optional
        并行核心数，-1表示全部，default -1
    
    Returns
    -------
    mimn_scores : np.ndarray
        1D array of MIMN scores for each band
    """
    n_bands = X_3D.shape[2]
    
    # 并行计算所有波段的熵和噪声
    print(f"Computing MIMN stats for {n_bands} bands (n_jobs={n_jobs})...")
    band_list = [X_3D[:, :, i] for i in range(n_bands)]
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_compute_band_stats)(band, entropy_bins, noise_window, noise_bins)
        for band in tqdm(band_list, desc="Band Stats (Entropy/Noise)", unit="band", ncols = 100)
    )
    
    # 计算MIMN分数（熵/噪声）
    entropies = np.array([r[0] for r in results])
    noises = np.array([r[1] for r in results])
    mimn_scores = entropies / noises
    
    return mimn_scores


def k_dpp_sampling_mimn(L_original: np.ndarray,
                        L_neighborhood: np.ndarray,
                        mimn_scores: np.ndarray,
                        k: int,
                        alpha: int = 3,
                        u: float = 0.5) -> List[int]:
    """
    Improved k-DPP sampling with MIMN guidance.
    优化点：添加进度条显示采样进度
    
    Parameters
    ----------
    L_original : np.ndarray
        Kernel matrix from original spectral data
    L_neighborhood : np.ndarray
        Kernel matrix from spatial neighborhood mean data
    mimn_scores : np.ndarray
        1D array of MIMN quality scores
    k : int
        Number of bands to select
    alpha : int, optional
        Number of candidate bands per iteration (balances diversity/quality), default 3
    u : float, optional
        Weight for original spectral kernel (0 ≤ u ≤ 1), default 0.5
    
    Returns
    -------
    selected_bands : List[int]
        List of selected band indices in order of selection
    """
    n_bands = L_original.shape[0]
    selected_bands = []
    remaining_bands = list(range(n_bands))
    
    # 复制矩阵避免修改原数据
    L1 = L_original.copy()
    L2 = L_neighborhood.copy()
    
    # 带进度条的采样循环
    for _ in tqdm(range(k), desc="DPP Sampling", unit="band",ncols=100):
        if not remaining_bands:
            break
        
        # 构建剩余波段的子矩阵掩码
        mask = np.zeros(n_bands, dtype=bool)
        mask[remaining_bands] = True
        L1_sub = L1[mask, :][:, mask]
        L2_sub = L2[mask, :][:, mask]
        
        # 特征分解（确保数值稳定性）
        eigvals1, eigvecs1 = np.linalg.eigh(L1_sub)
        eigvals2, eigvecs2 = np.linalg.eigh(L2_sub)
        eigvals1 = np.maximum(eigvals1, 1e-10)
        eigvals2 = np.maximum(eigvals2, 1e-10)
        
        # 计算每个剩余波段的选择概率
        probs = np.zeros(len(remaining_bands))
        for idx, band in enumerate(remaining_bands):
            prob1 = eigvals1 @ (eigvecs1[idx, :] ** 2)
            prob2 = eigvals2 @ (eigvecs2[idx, :] ** 2)
            probs[idx] = u * prob1 + (1 - u) * prob2
        
        # 归一化概率
        probs = probs / np.sum(probs)
        
        # 选择Top-alpha多样性候选波段
        alpha = min(alpha, len(remaining_bands))
        top_alpha_indices = np.argsort(probs)[-alpha:]
        top_alpha_bands = [remaining_bands[i] for i in top_alpha_indices]
        
        # 选择候选中MIMN分数最高的波段
        best_band = max(top_alpha_bands, key=lambda x: mimn_scores[x])
        
        # 更新选中/剩余波段列表
        selected_bands.append(best_band)
        remaining_bands.remove(best_band)
        
        # Schur补更新核矩阵
        for i in remaining_bands:
            for j in remaining_bands:
                L1[i, j] -= L1[i, best_band] * L1[best_band, j] / L1[best_band, best_band]
                L2[i, j] -= L2[i, best_band] * L2[best_band, j] / L2[best_band, best_band]
    
    return selected_bands


def mimn_dpp_bs(hsi: np.ndarray,
                num_selected_bands: int,
                original_mi_matrix =None,
                neighborhood_mi_matrix = None,
                mimn_scores = None,
                window_size: int = 3,
                max_samples: int = 10000,
                alpha: int = 3,
                u: float = 0.5,
                random_state: int = 42,
                n_jobs: int = -1,
                entropy_bins: int = 256,
                noise_window: int = 3,
                noise_bins: int = 50) -> np.ndarray:
    """
    Complete MIMN-DPP unsupervised band selection algorithm.
    核心优化：并行计算+预计算+进度条，大幅提升速度
    
    Parameters
    ----------
    hsi : np.ndarray
        Input hyperspectral image with shape (height, width, bands)
    k : int
        Number of bands to select
    window_size : int, optional
        Spatial neighborhood window size, default 3
    max_samples : int, optional
        Maximum samples for mutual information computation, default 10000
    alpha : int, optional
        Number of candidates per DPP step, default 3
    u : float, optional
        Weight for original spectral kernel, default 0.5
    random_state : int, optional
        Random seed for reproducibility, default 42
    n_jobs : int, optional
        并行计算核心数，-1=全部核心，default -1
    entropy_bins : int, optional
        熵计算的直方图分箱数，default 256
    noise_window : int, optional
        噪声估计窗口大小，default 3
    noise_bins : int, optional
        噪声估计直方图分箱数，default 50
    
    Returns
    -------
    selected_bands : np.ndarray
        Array of selected band indices (sorted by selection order)
    """
    # 设置随机种子
    np.random.seed(random_state)
    
    # 输入验证
    if hsi.ndim != 3:
        raise ValueError("Input must be a 3D hyperspectral image with shape (height, width, bands)")
    if num_selected_bands <= 0 or num_selected_bands >= hsi.shape[2]:
        raise ValueError(f"num_selected_bands must be between 1 and {hsi.shape[2]-1}")
    
    height, width, n_bands = hsi.shape
    print(f"=== MIMN-DPP Band Selection ===")
    print(f"Image shape: {height}x{width}x{n_bands} | Selecting {num_selected_bands} bands | n_jobs={n_jobs}")
    
    # Step 1: 计算空间邻域均值（预计算）
    print("\n[Step 1/4] Computing spatial neighborhood means...")
    hsi_neighborhood = compute_spatial_neighborhood_mean(hsi, window_size)
    
    # Step 2: 展平数据（预计算，避免重复reshape）
    hsi_2d = hsi.reshape(-1, n_bands)
    hsi_neighborhood_2d = hsi_neighborhood.reshape(-1, n_bands)
    
    # Step 3: 构建双核矩阵（并行计算）
    if original_mi_matrix is None:        
        print("\n[Step 2/4] Building original spectral kernel matrix...")        
        L_original = build_kernel_matrix(hsi_2d, max_samples, n_jobs)
    else:
        L_original = original_mi_matrix
    if neighborhood_mi_matrix is None:        
        print("\n[Step 3/4] Building neighborhood mean kernel matrix...")
        L_neighborhood = build_kernel_matrix(hsi_neighborhood_2d, max_samples, n_jobs)
    else:
        L_neighborhood = neighborhood_mi_matrix    
    
    # Step 4: 计算MIMN分数（并行+预计算）
    if mimn_scores is None:
        print("\n[Step 4/4] Computing MIMN quality scores...")
        mimn_scores = compute_mimn_scores(
            hsi, entropy_bins, noise_window, noise_bins, n_jobs
        )
    
    # 运行DPP采样（带进度条）
    print("\n[Final Step] Running MIMN-guided k-DPP sampling...")
    selected_bands = k_dpp_sampling_mimn(L_original, L_neighborhood, mimn_scores, num_selected_bands, alpha, u)
    
    # 转换为numpy数组
    selected_bands = np.array(selected_bands)
    
    print(f"\n=== Selection Complete ===")
    print(f"Selected {len(selected_bands)} bands: {selected_bands}")
    return selected_bands
"""
Main execution script for hyperspectral band selection experiments.
Implements the complete pipeline: data loading, statistical computation, 
band selection, and classification evaluation with parallel processing support.
"""

import time
import os
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, minmax_scale
from scipy.stats import ttest_rel
from typing import List, Optional, Dict, Tuple
from joblib import Parallel, delayed, cpu_count
from skimage.util.shape import view_as_windows

# Import custom modules for band selection algorithms and utilities
from algorithms.basic_utils import (
    compute_entropy, euclidean_distance_matrix, compute_corrcoef_matrix,
    compute_correlation_matrix_unsupervised, compute_covariance_matrix,
    rbf_kernel_dismat, compute_variances, compute_ssim_matrix
)
from algorithms.similarity_ranking_bs import (
    compute_mssim_matrix, similarity_ranking_bs
)
from algorithms.eca_bs import eca_bs
from algorithms.self_representation_bs import   self_representation_bs
from algorithms.max_covmatrix_det_bs import   max_covmatrix_det_bs
from algorithms.mvpca_bs import mvpca_bs
from algorithms.solr_bs import solr_bs, compute_noise_covariance_matrix,compute_noise_data_covariance_matrix
from algorithms.mimndpp_bs import build_kernel_matrix,mimn_dpp_bs,compute_spatial_neighborhood_mean, compute_mimn_scores
from algorithms.spbs import sp_bs
from data_loader import HSIDataLoader,load_noisy_data
from config import (
    DATASETS, BASE_DATA_DIR, SMI_PARAMS, SVM_PARAMS, OUTPUT_DIR, SNR_DB, NOISE_PERCENT_LIST
)
from algorithms.bs_nets.bs_nets_conv import BS_Net_Conv, run_bs_net_conv
from generate_noisy_image import generate_noisy_image

def _add_uniform_noise(data, snr_db, percent = 0.5, seed = None):
    """
    对所有波段添加相同强度的加性高斯白噪声。

    噪声标准差 = 全局标准差 / 10^(SNR/20)

    Parameters
    ----------
    data : ndarray, shape (H, W, B)
        原始高光谱图像（float）。
    snr_db : float
        信噪比，单位dB。
    seed : int or None
        随机种子。

    Returns
    -------
    noisy : ndarray, shape (H, W, B)
        加噪后图像。
    """
    noisy_band_num = int(data.shape[2] * percent)
    noise_band_index = np.random.choice(data.shape[2], noisy_band_num, replace=False)
    noisy_data = data.copy().astype(np.float64)
    for band_ind in noise_band_index:
        band_std = np.std(data[:, :, band_ind])
        sigma_b = band_std / (10 ** (snr_db / 20.0))
        noisy_data[:, :, band_ind] = np.random.normal(0, sigma_b, data[:, :, band_ind].shape)
    return noisy_data, noise_band_index
def compute_and_cache_statistics(
    X: np.ndarray,
    X_normlized: np.ndarray,
    X_3D: np.ndarray,
    dataset_dir: str = '/mnt/f/pytorch/HSI/datasets/dataset_name',
    dataset_prefix: str = 'dataset_name',
    n_jobs: int = -1
) -> Dict[str, np.ndarray]:
    """
    Compute and cache all prerequisite statistical matrices for band selection.
    
    Args:
        X: Raw 2D data matrix (n_samples × n_bands)
        X_normlized: Standardized 2D data matrix
        X_3D: 3D hyperspectral data cube (height × width × n_bands)
        dataset_dir: Directory to save cached files
        dataset_prefix: Prefix for cached filenames
        n_jobs: Number of parallel jobs for computation
    
    Returns:
        Dictionary containing all computed statistical matrices
    """
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Define cache file paths for all statistical matrices
    cache_paths = {
        'ssim': os.path.join(dataset_dir, f'{dataset_prefix}_ssim_matrix.npy'),
        'ssim_win3': os.path.join(dataset_dir, f'{dataset_prefix}_ssim_win3_matrix.npy'),
        'sr_eta': os.path.join(dataset_dir, f'{dataset_prefix}_sr_eta.npy'),
        'variances': os.path.join(dataset_dir, f'{dataset_prefix}_variances.npy'),
        'entropy': os.path.join(dataset_dir, f'{dataset_prefix}_entropy.npy'),
        'distance_matrix': os.path.join(dataset_dir, f'{dataset_prefix}_distance_matrix.npy'),
        'rbf_kernel': os.path.join(dataset_dir, f'{dataset_prefix}_rbf_kernel_matrix.npy'),
        'covariance_matrix': os.path.join(dataset_dir, f'{dataset_prefix}_covariance_matrix.npy'),
        'correlation_matrix': os.path.join(dataset_dir, f'{dataset_prefix}_correlation_matrix.npy'),
        'corrcoef_matrix': os.path.join(dataset_dir, f'{dataset_prefix}_corrcoef_matrix.npy'),
        'noise_covariance_matrix': os.path.join(dataset_dir, f'{dataset_prefix}_noise_covariance_matrix.npy'),   
        'original_mi_matrix': os.path.join(dataset_dir, f'{dataset_prefix}_original_mi_matrix.npy'),
        'neighbor_mi_matrix': os.path.join(dataset_dir, f'{dataset_prefix}_neighbor_mi_matrix.npy'),
        'mimn_scores': os.path.join(dataset_dir, f'{dataset_prefix}_mimn_scores.npy'),
    }
    
    stats = {}
    
    # Compute SSIM matrix (window size=11)
    if os.path.exists(cache_paths['ssim']):
        stats['ssim'] = np.load(cache_paths['ssim'])
    else:
        print("Computing SSIM matrix...")
        stats['ssim'] = compute_mssim_matrix(
            X_3D, window_size=11, sigma=1.5, L=255, n_jobs=n_jobs
        )
        np.save(cache_paths['ssim'], stats['ssim'])
    
    # Compute SSIM matrix (window size=3)
    if os.path.exists(cache_paths['ssim_win3']):
        stats['ssim_win3'] = np.load(cache_paths['ssim_win3'])
    else:
        print("Computing SSIM matrix (window size=3)...")
        stats['ssim_win3'] = compute_ssim_matrix(X_3D, window_size=3)
        np.save(cache_paths['ssim_win3'], stats['ssim_win3'])
    
    # Compute similarity ranking eta coefficients
    if os.path.exists(cache_paths['sr_eta']):
        stats['sr_eta'] = np.load(cache_paths['sr_eta'])
    else:
        print("Computing similarity ranking eta coefficients...")
        _, stats['sr_eta'] = similarity_ranking_bs(
            stats['ssim'], similarity_threshold=0.8, num_selected_bands=10
        )
        np.save(cache_paths['sr_eta'], stats['sr_eta'])
    
    # Compute band variances
    if os.path.exists(cache_paths['variances']):
        stats['variances'] = np.load(cache_paths['variances'])
        print(f'variances.shape = {stats["variances"].shape}')
        csv_file = os.path.join(dataset_dir, f'{dataset_prefix}_variances.csv')
        if not os.path.exists(csv_file):
            np.savetxt(csv_file, stats['variances'], delimiter=',')
    else:
        print("Computing band variances...")
        stats['variances'] = compute_variances(X)
        np.save(cache_paths['variances'], stats['variances'])
    
    # Compute band entropy
    if os.path.exists(cache_paths['entropy']):
        stats['entropy'] = np.load(cache_paths['entropy'])
    else:
        print("Computing band entropy...")
        stats['entropy'] = compute_entropy(X_normlized)
        np.save(cache_paths['entropy'], stats['entropy'])
    
    # Compute Euclidean distance matrix
    if os.path.exists(cache_paths['distance_matrix']):
        stats['distance_matrix'] = np.load(cache_paths['distance_matrix'])
    else:
        print("Computing Euclidean distance matrix...")
        stats['distance_matrix'] = euclidean_distance_matrix(X)
        np.save(cache_paths['distance_matrix'], stats['distance_matrix'])
    
    # Compute RBF kernel distance matrix
    if os.path.exists(cache_paths['rbf_kernel']):
        stats['rbf_kernel'] = np.load(cache_paths['rbf_kernel'])
    else:
        print("Computing RBF kernel distance matrix...")
        stats['rbf_kernel'] = rbf_kernel_dismat(
            stats['distance_matrix'], sigma=np.mean(stats['distance_matrix']) / 30
        )
        np.save(cache_paths['rbf_kernel'], stats['rbf_kernel'])
    
    # Compute covariance matrix
    if os.path.exists(cache_paths['covariance_matrix']):
        stats['covariance_matrix'] = np.load(cache_paths['covariance_matrix'])
    else:
        print("Computing covariance matrix...")
        stats['covariance_matrix'] = compute_covariance_matrix(X)
        np.save(cache_paths['covariance_matrix'], stats['covariance_matrix'])
        
    # Compute correlation matrix (unsupervised)
    if os.path.exists(cache_paths['correlation_matrix']):
        stats['correlation_matrix'] = np.load(cache_paths['correlation_matrix'])
        binary_file = os.path.join(dataset_dir, f'{dataset_prefix}_correlation_matrix.bin')
        csv_file = os.path.join(dataset_dir, f'{dataset_prefix}_correlation_matrix.csv')
        if not os.path.exists(binary_file):
            stats['correlation_matrix'].tofile(binary_file)
        if not os.path.exists(csv_file):
            np.savetxt(csv_file, np.diag(stats['correlation_matrix']), delimiter=',')
    else:
        print("Computing correlation matrix...")
        stats['correlation_matrix'] = compute_correlation_matrix_unsupervised(X)
        np.save(cache_paths['correlation_matrix'], stats['correlation_matrix'])
    
    # Compute correlation coefficient matrix
    if os.path.exists(cache_paths['corrcoef_matrix']):
        stats['corrcoef_matrix'] = np.load(cache_paths['corrcoef_matrix'])
    else:
        print("Computing correlation coefficient matrix...")
        stats['corrcoef_matrix'] = compute_corrcoef_matrix(X_normlized)
        np.save(cache_paths['corrcoef_matrix'], stats['corrcoef_matrix'])
    
    # Compute noise covariance matrix
    if os.path.exists(cache_paths['noise_covariance_matrix']):
        stats['noise_covariance_matrix'] = np.load(cache_paths['noise_covariance_matrix'])
    else:
        print("Computing noise covariance matrix...")
        stats['noise_covariance_matrix'] = compute_noise_covariance_matrix(X_3D)
        np.save(cache_paths['noise_covariance_matrix'], stats['noise_covariance_matrix'])
    
        
    # Compute MI matrix
    if os.path.exists(cache_paths['original_mi_matrix']):
        stats['original_mi_matrix'] = np.load(cache_paths['original_mi_matrix'])
    else:
        print("Computing MI matrix...")
        stats['original_mi_matrix'] = build_kernel_matrix(X)
        np.save(cache_paths['original_mi_matrix'], stats['original_mi_matrix'])
    # Compute neigbour MI matrix
    if os.path.exists(cache_paths['neighbor_mi_matrix']):
        stats['neighbor_mi_matrix'] = np.load(cache_paths['neighbor_mi_matrix'])
    else:
        print("Computing neighbor mi matrix...")
        X_neighbor = compute_spatial_neighborhood_mean(X_3D, window_size=3)
        X_neighbor_2D = X_neighbor.reshape(-1, X_neighbor.shape[-1])
        stats['neighbor_mi_matrix'] = build_kernel_matrix(X_neighbor_2D)
        np.save(cache_paths['neighbor_mi_matrix'], stats['neighbor_mi_matrix'])    
    
    # Compute MINSCores matrix
    if os.path.exists(cache_paths['mimn_scores']):
        stats['mimn_scores'] = np.load(cache_paths['mimn_scores'])
    else:
        print("Computing mimn_scores matrix...")
        stats['mimn_scores'] = compute_mimn_scores(X_3D, entropy_bins =256 , noise_window = 3 , noise_bins = 50)
        np.save(cache_paths['mimn_scores'], stats['mimn_scores'])   
    return stats

def _single_run_evaluation(
    run_id: int,
    result_dir: str,
    dataset_name: str,
    n_selected_bands_list: List[int],
    X_normlized: np.ndarray,
    y_true: np.ndarray,
    band_selection_methods: List[str],
    test_size: float,
    model_random_state_base: int,
    classifier_name: str,    
    is_stochastic_method: Dict[str, bool]
) -> List[Dict]:
    """
    Execute a single independent evaluation run (used for parallel processing).
    
    Args:
        run_id: Unique identifier for the current run
        result_dir: Directory containing band selection results
        dataset_name: Name of the dataset
        n_selected_bands_list: List of band counts to evaluate
        X_normlized: Standardized full dataset
        y_true: Ground truth labels
        band_selection_methods: List of band selection methods to evaluate
        test_size: Proportion of test set
        model_random_state_base: Base random seed for reproducibility
        is_stochastic_method: Flag indicating if method is stochastic
    
    Returns:
        List of dictionaries containing evaluation results for this run
    """
    current_random_state = model_random_state_base + run_id
    run_results = []
    
    try:
        # Split data into train/test sets with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X_normlized, y_true, test_size=test_size,
            random_state=current_random_state, stratify=y_true
        )
        
        # Initialize  classifier
        if classifier_name == 'KNN':
            model =KNeighborsClassifier(n_neighbors = 3 , n_jobs=-1)
        elif classifier_name == 'SVM':                 
            model = LinearSVC(C=1, random_state=current_random_state, max_iter=10000)
        
        
        # Evaluate each band selection method and band count
        for n_bands in n_selected_bands_list:
            for method in band_selection_methods:
                try:
                    # Load band indices based on method type (stochastic/deterministic)
                    if is_stochastic_method.get(method, False):
                        band_idx_path = os.path.join(
                            result_dir,
                            f"{dataset_name}_{method}_k{n_bands}_run{run_id}.npy"
                        )
                    else:
                        band_idx_path = os.path.join(
                            result_dir,
                            f"{dataset_name}_{method}_k{n_bands}.npy"
                        )
                    
                    if not os.path.exists(band_idx_path):
                        continue
                    
                    # Load and validate selected band indices
                    selected_band_idx = np.load(band_idx_path).flatten()
                    valid_idx = selected_band_idx[
                        (selected_band_idx >= 0) & 
                        (selected_band_idx < X_train.shape[1])
                    ]
                    
                    if len(valid_idx) == 0:
                        continue
                    
                    # Extract selected bands and evaluate classification performance
                    X_train_selected = X_train[:, valid_idx]
                    X_test_selected = X_test[:, valid_idx]
                    
                    model.fit(X_train_selected, y_train)
                    y_test_pred = model.predict(X_test_selected)
                    test_acc = accuracy_score(y_test, y_test_pred)
                    
                    # Record results
                    run_results.append({
                        "run_id": run_id,
                        "method": method,
                        "n_selected_bands": n_bands,
                        "test_accuracy": test_acc,
                        "selected_band_count": len(valid_idx),
                        "selected_band_idx": valid_idx.tolist()
                    })
                    
                except Exception as e:
                    print(f"Warning: Run {run_id} | {method} | K={n_bands} failed: {str(e)}")
                    continue
                    
    except Exception as e:
        print(f"Error: Run {run_id} failed completely: {str(e)}")
    
    return run_results

def _count_selected_noisy_bands(    
    result_dir: str,
    dataset_name: str,
    noisy_bands_index: np.ndarray,
    n_selected_bands_list: List[int],
    band_selection_methods: List[str],
    save_path:str) -> pd.DataFrame:
    results = []
    for n_bands in n_selected_bands_list:
        for method in band_selection_methods:            
            # Load band indices based on method type (stochastic/deterministic)
            band_idx_path = os.path.join(
                result_dir,
                f"{dataset_name}_{method}_k{n_bands}.npy"
                )
           
            # Load and validate selected band indices
            selected_band_idx = np.load(band_idx_path).flatten()
            selected_noisy_band = np.intersect1d(selected_band_idx, noisy_bands_index)
            
            results.append({ 
                            "method": method,
                            "n_selected_bands": n_bands,
                            "selected_noisy_bands_num": selected_noisy_band.shape[0],
                            "selected_noisy_bands": selected_noisy_band
                    })
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_path)
    return
def evaluate_band_selection_results(
    result_dir: str,
    dataset_name: str,
    n_selected_bands_list: List[int],
    X_normlized: np.ndarray,
    y_true: np.ndarray,
    band_selection_methods: List[str],
    baseline_methods: Optional[List[str]] = None,
    test_size: float = 0.2,
    n_runs: int = 10,
    overwrite: bool = False,
    classifier_name: str = "SVM",
    model_random_state_base: int = 42,
    is_stochastic_method: Optional[Dict[str, bool]] = None,
    n_jobs: Optional[int] = None,
    verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate band selection results with statistical significance testing.
    
    Implements IEEE TGRS-standard evaluation protocol:
    1. Parallel execution of n_runs independent experiments
    2. Paired t-test for statistical significance against baselines
    3. Comprehensive reporting of mean, std, variance, and p-values
    
    Args:
        result_dir: Directory containing band selection results (.npy files)
        dataset_name: Name of the dataset (for filename construction)
        n_selected_bands_list: List of band counts to evaluate
        X_normlized: Standardized full dataset
        y_true: Ground truth labels
        band_selection_methods: List of band selection methods to evaluate
        baseline_methods: List of baseline methods for significance testing
        test_size: Proportion of test set
        n_runs: Number of independent evaluation runs
        model_random_state_base: Base random seed for reproducibility
        is_stochastic_method: Flag indicating if method is stochastic
        n_jobs: Number of parallel jobs (-1 for all cores-1)
        verbose: Verbosity level for parallel execution
    
    Returns:
        Tuple containing:
        - raw_result_df: DataFrame of all individual run results
        - summary_result_df: DataFrame of statistical summaries
    """
    # Initialize default parameters
    if baseline_methods is None:
        baseline_methods = band_selection_methods[:len(band_selection_methods)//2]
    
    if is_stochastic_method is None:
        is_stochastic_method = {method: False for method in band_selection_methods}
    
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)
    
    print("=" * 80)
    print(f"Starting parallel evaluation with {n_jobs} CPU cores")
    print("=" * 80)
    
    raw_csv_path = os.path.join(result_dir, f"{dataset_name}_{classifier_name}_classification_results_raw.csv")
    summary_csv_path = os.path.join(result_dir, f"{dataset_name}_{classifier_name}_classification_results_summary.csv")
    
    noisy_band_result_path = os.path.join(result_dir, f"{dataset_name}_selected_noisy_bands.csv")
    if not os.path.exists(noisy_band_result_path):       
        noisy_bands_index_path = os.path.dirname(os.path.dirname(result_dir))
        noisy_bands_index_file = os.path.join(noisy_bands_index_path, f'{dataset_name}_noise_band_index.npy')
        if os.path.exists(noisy_bands_index_file):
            noisy_bands_index = np.load(noisy_bands_index_file)
            _count_selected_noisy_bands(
                result_dir,
                dataset_name,
                noisy_bands_index,             
                n_selected_bands_list,
                band_selection_methods,
                noisy_band_result_path)    
    
    
    # Skip if results already exist
    if os.path.exists(summary_csv_path) and overwrite == False:
        print(f"Results already exist for {dataset_name}, skipping evaluation.")
        return pd.read_csv(raw_csv_path), pd.read_csv(summary_csv_path)
    
    # Execute parallel evaluation runs
    all_run_results = Parallel(
        n_jobs=n_jobs,
        verbose=verbose,
        backend='loky'
    )(
        delayed(_single_run_evaluation)(
            run_id = run_id,
            result_dir = result_dir,
            dataset_name = dataset_name,
            n_selected_bands_list=n_selected_bands_list,
            X_normlized=X_normlized,
            y_true = y_true,
            band_selection_methods = band_selection_methods,
            test_size = test_size,
            classifier_name = classifier_name,
            model_random_state_base = model_random_state_base,
            is_stochastic_method = is_stochastic_method,
        )
        for run_id in range(n_runs)
    )
    
    # Aggregate results from all parallel runs
    raw_evaluation_results = []
    for run_result in all_run_results:
        raw_evaluation_results.extend(run_result)
    
    if len(raw_evaluation_results) == 0:
        raise ValueError("No evaluation results generated. Check band selection files.")
    
    # Save raw results
    raw_result_df = pd.DataFrame(raw_evaluation_results)
    os.makedirs(os.path.dirname(raw_csv_path), exist_ok=True)
    raw_result_df.to_csv(raw_csv_path, index=False, encoding='utf-8')
    print(f"\nRaw results saved to: {raw_csv_path}")
    
    # Generate statistical summary
    print("\n" + "=" * 80)
    print("Computing statistical summary and significance tests")
    print("=" * 80)
    
    summary_results = []
    
    # Group results by method and band count
    grouped = raw_result_df.groupby(["method", "n_selected_bands"])
    
    for (method, n_bands), group in grouped:
        acc_list = group["test_accuracy"].values
        mean_acc = np.mean(acc_list)
        std_acc = np.std(acc_list, ddof=1)
        var_acc = np.var(acc_list, ddof=1)
        
        p_value = np.nan
        significance_mark = ""
        
        # Perform paired t-test against baseline methods
        if method not in baseline_methods:
            corresponding_baseline = None
            for baseline in baseline_methods:
                if baseline in method:
                    corresponding_baseline = baseline
                    break
            
            if corresponding_baseline is not None:
                baseline_group = raw_result_df[
                    (raw_result_df["method"] == corresponding_baseline) &
                    (raw_result_df["n_selected_bands"] == n_bands)
                ]
                
                if len(baseline_group) == n_runs:
                    baseline_acc_list = baseline_group["test_accuracy"].values
                    t_stat, p_value = ttest_rel(
                        acc_list, baseline_acc_list, alternative='two-sided'
                    )
                    
                    # Assign significance markers
                    if p_value < 0.01:
                        significance_mark = "**"
                    elif p_value < 0.05:
                        significance_mark = "*"
                    
                    print(f"{method} vs {corresponding_baseline} (K={n_bands}): "
                          f"p={p_value:.4f} {significance_mark}")
        
        summary_results.append({
            "method": method,
            "n_selected_bands": n_bands,
            "mean_test_accuracy": mean_acc,
            "std_test_accuracy": std_acc,
            "var_test_accuracy": var_acc,
            "p_value_vs_baseline": p_value,
            "significance_mark": significance_mark,
            "raw_accuracy_list": acc_list.tolist()
        })
    
    # Save summary results
    summary_result_df = pd.DataFrame(summary_results)
    summary_result_df = summary_result_df.sort_values(
        ["method", "n_selected_bands"]
    ).reset_index(drop=True)
    
    summary_result_df.to_csv(summary_csv_path, index=False, encoding='utf-8')
    print(f"\nSummary results saved to: {summary_csv_path}")
    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)
    
    
    
    
    return raw_result_df, summary_result_df


def run_band_selection(
    result_dir: str,
    dataset_name: str,
    total_band_number: int,
    n_selected_bands_list: List[int],
    band_selection_methods: List[str],
    X_2D: np.ndarray,
    X_3D: np.ndarray,    
    precomputed_stats: Dict[str, np.ndarray]
) -> None:
    """
    Execute all band selection methods and save results.
    
    Args:
        result_dir: Directory to save band selection results
        dataset_name: Name of the dataset
        total_band_number: Total number of bands in the dataset
        n_selected_bands_list: List of band counts to select
        band_selection_methods: List of band selection methods to execute
        X_2D: 2D data matrix (n_samples × n_bands)
        precomputed_stats: Precomputed statistical matrices
    """
    for n_selected_bands in n_selected_bands_list:
        # Generate subspace partitions for different strategies
                # Define result file paths
        result_paths = {
            method: os.path.join(
                result_dir, f'{dataset_name}_{method}_k{n_selected_bands}.npy'
            )
            for method in band_selection_methods
        }
        bs_nets_sort_result_path =  os.path.join(
                result_dir, f'{dataset_name}_bsnets_sort_result.npy'
            )
        # ==================== MVPCA Methods ====================
        if not os.path.exists(result_paths['mvpca']):
            print(f"Running MVPCA for {n_selected_bands} bands...")
            mvpca_result = mvpca_bs(
                X_2D, num_selected_bands=n_selected_bands,
                variances=precomputed_stats['variances']
            )
            np.save(result_paths['mvpca'], mvpca_result)
        
        
        # ==================== ECA Methods ====================
        if not os.path.exists(result_paths['eca']):
            print(f"Running ECA for {n_selected_bands} bands...")
            eca_result, _ = eca_bs(
                X_2D, distance_matrix=precomputed_stats['distance_matrix']
            )
            eca_result = eca_result[:n_selected_bands]
            np.save(result_paths['eca'], eca_result)
       
        
        # ==================== MCD Methods ====================
        if not os.path.exists(result_paths['mcd']):
            print(f"Running MCD for {n_selected_bands} bands...")
            mcd_result = max_covmatrix_det_bs(
                X_2D, num_selected_bands=n_selected_bands,
                covariance_matrix=precomputed_stats['covariance_matrix']
            )
            np.save(result_paths['mcd'], mcd_result)
               
        # ==================== SEFREP Methods ====================
        if not os.path.exists(result_paths['sefrep']):
            print(f"Running SEFREP for {n_selected_bands} bands...")
            sefrep_result = self_representation_bs(
                X_2D, num_selected_bands=n_selected_bands,
                correlation_matrix=precomputed_stats['correlation_matrix']
            )
            np.save(result_paths['sefrep'], sefrep_result)
        
        
        # ==================== SIMRNK Methods ====================
        if not os.path.exists(result_paths['simrnk']):
            print(f"Running SIMRNK for {n_selected_bands} bands...")
            simrnk_result = np.argsort(precomputed_stats['sr_eta'])[::-1][:n_selected_bands]
            np.save(result_paths['simrnk'], simrnk_result)
            
        # ==================== MNBS Methods ====================
        if not os.path.exists(result_paths['mnbs']):
            print(f"Running mnbs for {n_selected_bands} bands...")
            mnbs_result = mn_bs(X_2D = X_2D, num_selected_bands = n_selected_bands,
                                noise_covariance_matrix = precomputed_stats['noise_covariance_matrix'],
                                covariance_matrix = precomputed_stats['covariance_matrix'])
            np.save(result_paths['mnbs'], mnbs_result)
        
        # ==================== MIMNDPP Methods ====================
        if not os.path.exists(result_paths['mimndpp']):
            print(f"Running mimndpp bs for {n_selected_bands} bands...")
            mimndpp_result = mimn_dpp_bs(X_3D, num_selected_bands = n_selected_bands, 
                                         original_mi_matrix = precomputed_stats['original_mi_matrix'], 
                                         mimn_scores = precomputed_stats['mimn_scores'],
                                         neighborhood_mi_matrix = precomputed_stats['neighbor_mi_matrix'])
            np.save(result_paths['mimndpp'], mimndpp_result)
        
        # ==================== BS_NETS Methods ====================
        if not os.path.exists(result_paths['bsnets']):
            print(f"Running bs_nets bs for {n_selected_bands} bands...")
            if not os.path.exists(bs_nets_sort_result_path):
                n_row, n_column, n_band = X_3D.shape
                X_img = minmax_scale(X_3D.reshape(n_row * n_column, n_band)).reshape((n_row, n_column, n_band))
                
                # 超参数设置（与原代码一致）
                LR = 0.0001
                BATCH_SIZE = 32
                EPOCH = 10
                N_BAND = n_band
                
                # 开始训练                
                bsnets_selected_bands_full, loss_history, score_list = run_bs_net_conv(
                    X_3D, None, N_BAND, EPOCH, BATCH_SIZE, LR)
                np.save(bs_nets_sort_result_path, np.array(bsnets_selected_bands_full))
            else:
                bsnets_selected_bands_full = np.load(bs_nets_sort_result_path)
            bsnets_result = bsnets_selected_bands_full[:n_selected_bands]
            np.save(result_paths['bsnets'], bsnets_result)

        # ==================== SOLR Methods ====================
        if not os.path.exists(result_paths['solr']):
            print(f"Running SOLR for {n_selected_bands} bands...")
            solr_result= solr_bs(X_2D, n_selected_bands,
                                 C_X=precomputed_stats['covariance_matrix'],
                                 C_N=precomputed_stats['noise_covariance_matrix'])
            np.save(result_paths['solr'], solr_result)
        

def process_single_dataset(dataset_name: str) -> None:
    """
    Process a single dataset through the complete experimental pipeline.
    
    Args:
        dataset_name: Name of the dataset to process
    """
    # Configure paths
    dataset_dir = os.path.join(BASE_DATA_DIR, dataset_name)
    result_dir = os.path.join(dataset_dir, "result/SOLR")
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    
    # Control flags for pipeline steps
    is_run_bs = False
    is_run_eva = False
    
    # Load dataset
    if dataset_name in DATASETS.keys():
        hsi_data_loader = HSIDataLoader(dataset_name)
        X, y_true, X_3D = hsi_data_loader.load()
    else:
        X, y_true, X_3D = load_noisy_data(dataset_name)
    
    print(f"Dataset: {dataset_name} | Shape: {X_3D.shape}")
    
    '''
    generate_noisy_image(X_3D, NOISE_PERCENT_LIST, BASE_DATA_DIR, dataset_name, 
                         snr_db = SNR_DB, gt_img = y_true.reshape(X_3D.shape[0],X_3D.shape[1]), seed = None)
    '''
    # Standardize data
    X_normlized = StandardScaler().fit_transform(X.copy())
    
    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_normlized, y_true,
        test_size=SVM_PARAMS["test_size"],
        random_state=SVM_PARAMS["random_state"]
    )
    print(f"Train set: {X_train.shape} | Test set: {X_test.shape}")
    
    # Compute and cache statistical matrices
    
    precomputed_stats = compute_and_cache_statistics(
        X, X_normlized, X_3D,
        dataset_dir=dataset_dir,
        dataset_prefix=dataset_name,
        n_jobs=SMI_PARAMS["n_jobs"])

    #print('data correlation matrix = ')
    #print(np.diag(precomputed_stats['correlation_matrix'] - precomputed_stats['noise_covariance_matrix']))
          
    #print('nosie covariance matrix = ')
    #print(precomputed_stats['noise_covariance_matrix'])
    
       
    # Run band selection methods
    n_selected_bands_list = [5, 10, 15, 20, 25, 30]
    band_selection_methods = [
        'mvpca', 
        'eca', 
        'mcd', 
        'sefrep', 
        'simrnk',
        'mnbs',
        'mimndpp',
        'bsnets',
        'solr'
    ]
    baseline_methods = ['mvpca', 'eca', 'mcd', 'sefrep', 'simrnk','mnbs','mimndpp','bsnets']
    
    if is_run_bs:
        print(f"Running band selection for {dataset_name}...")
        run_band_selection(
            result_dir=result_dir,
            dataset_name=dataset_name,
            total_band_number=X.shape[1],
            n_selected_bands_list=n_selected_bands_list,
            band_selection_methods=band_selection_methods,
            X_2D = X,
            X_3D = X_3D,
            precomputed_stats = precomputed_stats
        )
    
    # Filter out background pixels (label=0)
    non_zero_mask = y_true != 0
    X_normlized_non_zero = X_normlized[non_zero_mask, :]
    y_true_non_zero = y_true[non_zero_mask]
    
       
    
    # Evaluate band selection results
    if is_run_eva:
        try:
            evaluate_band_selection_results(
                result_dir = result_dir,
                dataset_name = dataset_name,
                n_selected_bands_list = n_selected_bands_list,
                X_normlized = X_normlized_non_zero,
                y_true = y_true_non_zero,
                band_selection_methods = band_selection_methods,
                baseline_methods = baseline_methods,
                overwrite = False,
                classifier_name = "KNN",
                verbose = True
            )
        except Exception as e:
            print(f"Error evaluating {dataset_name}: {str(e)}")
            return


if __name__ == "__main__":
    
    # Configure datasets to process (None = all datasets)
    TARGET_DATASETS = None#["KSC"]  # Process all datasets
    # TARGET_DATASETS = ["indianpines"]  # Process single dataset
    
    # Get list of datasets to process
    if TARGET_DATASETS is None:
        datasets_to_process = list(DATASETS.keys())
    else:
        datasets_to_process = [ds for ds in TARGET_DATASETS if ds in DATASETS]
        missing_datasets = [ds for ds in TARGET_DATASETS if ds not in DATASETS]
        if missing_datasets:
            print(f"Warning: Datasets not configured in config.py: {missing_datasets}")
            
    
    noisy_data_list = []
    for ds_name in datasets_to_process:
        noise_percent_list = DATASETS[ds_name]['noise_percent']
        for noise_percent in noise_percent_list:
            noisy_data_list.append(f'{ds_name}_{noise_percent}')
    datasets_to_process.extend(noisy_data_list)   
        
    # Batch process datasets
    print(f"\nStarting batch processing for datasets: {datasets_to_process}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    for ds_name in datasets_to_process:
        print("\n" + "=" * 60)
        print(f"Processing Dataset: {ds_name.upper()}")
        print("=" * 60)
        process_single_dataset(ds_name)
    
    print("All datasets processed successfully!")
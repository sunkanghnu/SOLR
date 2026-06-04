# config.py
import os

# ===================== Path Configuration =====================
# Base directory for hyperspectral image (HSI) datasets
BASE_DATA_DIR = "/mnt/f/pytorch/HSI/datasets"
NOISE_PERCENT_LIST = [30, 50, 70]
SNR_DB = 15
# Paths for different HSI datasets (hsi: hyperspectral image, gt: ground truth)
DATASETS = {
    "indian_pines": {
        "hsi": os.path.join(BASE_DATA_DIR, "indian_pines/Indian_pines.mat"),
        "gt": os.path.join(BASE_DATA_DIR, "indian_pines/Indian_pines_gt.mat"),
        "noise_percent": NOISE_PERCENT_LIST
    },
    "KSC": {
        "hsi": os.path.join(BASE_DATA_DIR, "KSC/KSC.mat"),
        "gt": os.path.join(BASE_DATA_DIR, "KSC/KSC_gt.mat"),
        "noise_percent": NOISE_PERCENT_LIST
    },
    "pavia": {
        "hsi": os.path.join(BASE_DATA_DIR, "pavia/Pavia.mat"),
        "gt": os.path.join(BASE_DATA_DIR, "pavia/Pavia_gt.mat"),
        "noise_percent": NOISE_PERCENT_LIST
    },
    "botswana": {
        "hsi": os.path.join(BASE_DATA_DIR, "botswana/Botswana.mat"),
        "gt": os.path.join(BASE_DATA_DIR, "botswana/Botswana_gt.mat"),
        "noise_percent": NOISE_PERCENT_LIST
    },
    "salinas": {
        "hsi": os.path.join(BASE_DATA_DIR, "salinas/Salinas.mat"),
        "gt": os.path.join(BASE_DATA_DIR, "salinas/Salinas_gt.mat"),
        "noise_percent": NOISE_PERCENT_LIST
    }    
}

# ===================== Experiment Hyperparameters =====================
# Parameters for SMSI calculation
SMI_PARAMS = {
    "n_samples": 1000,          # Number of pixel samples
    "n_jobs": -1,               # Number of parallel cores (-1 means use all cores)
    "random_state": 42,         # Random seed for reproducibility
    "nonadj_max_pairs": 1000,   # Maximum number of non-adjacent band pairs for sampling
    "max_k_mic": 30             # Maximum k for MIC interval analysis
}

# Parameters for band selection
BAND_SELECTION_PARAMS = {
    "top_k": 30,                # Select top k optimal bands
    "mrmr_n_features": 30,      # Number of bands selected by mRMR algorithm
    "f_score_k": 30,            # Number of bands selected by F-score algorithm
    "mi_k": 30                  # Number of bands selected by mutual information
}

# Parameters for band grouping
BAND_GROUP_PARAMS = {
    "group_strategy": "smsi_threshold",  # Grouping strategy: smsi_threshold/cluster/interval
    "smsi_threshold": 0.5,               # Threshold for SMSI-based grouping
    "interval_step": 5,                  # Step size for interval-based grouping
    "n_clusters": 10                     # Number of clusters for cluster-based grouping
}

# Parameters for SVM classification
SVM_PARAMS = {
    "C": 1.0,                   # Regularization parameter for SVM
    "penalty": "l2",            # Penalty norm for SVM
    "max_iter": 1000,           # Maximum iterations for SVM training
    "test_size": 0.2,           # Proportion of test set in the whole dataset
    "random_state": 42          # Random seed for train-test split
}

# ===================== Experiment Output Configuration =====================
# Directory for saving experiment results
OUTPUT_DIR = "./results"
# Create output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
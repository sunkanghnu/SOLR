# data_loader.py
import scipy.io
import numpy as np
import os
from config import DATASETS, BASE_DATA_DIR

class HSIDataLoader:
    """Hyperspectral Image (HSI) Data Loader
    Supports multiple datasets with optional preprocessing
    """
    def __init__(self, dataset_name):
        """Initialize the HSI data loader
        
        Args:
            dataset_name (str): Name of the dataset to load (e.g., "indian_pines", "pavia")
        """
        self.dataset_name = dataset_name
        self.hsi_path = DATASETS[self.dataset_name]["hsi"]
        self.gt_path = DATASETS[self.dataset_name]["gt"]
    
    def load(self):
        """Load dataset and return processed 2D data, labels, and original 3D data
        
        Returns:
            tuple: (2D data array, label array, original 3D HSI array)
        """
        if self.dataset_name == "indian_pines":
            return self._load_indianpines()
        elif self.dataset_name == "pavia":
            return self._load_pavia()
        elif self.dataset_name == "botswana":
            return self._load_botswana()
        elif self.dataset_name == "KSC":
            return self._load_ksc()
        elif self.dataset_name == "salinas":
            return self._load_salinas()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    
    def _load_indianpines(self):
        """Load Indian Pines dataset
        
        Returns:
            tuple: (2D data array, label array, original 3D HSI array)
        """
        hsi_data = scipy.io.loadmat(self.hsi_path)
        X_3D = hsi_data['indian_pines_corrected'].astype(np.float64)
        X_2D = X_3D.reshape(-1, X_3D.shape[-1])
        
        gt_data = scipy.io.loadmat(self.gt_path)
        y_true = gt_data['indian_pines_gt'].reshape(-1).T
        return X_2D, y_true, X_3D
    
    def _load_pavia(self):
        """Load Pavia dataset
        
        Returns:
            tuple: (2D data array, label array, original 3D HSI array)
        """
        hsi_data = scipy.io.loadmat(self.hsi_path)
        X_3D = hsi_data['pavia'].astype(np.float64)
        X_2D = X_3D.reshape(-1, X_3D.shape[-1])
        
        gt_data = scipy.io.loadmat(self.gt_path)
        y_true = gt_data['pavia_gt'].reshape(-1).T
        return X_2D, y_true, X_3D
    
    def _load_botswana(self):
        """Load Botswana dataset
        
        Returns:
            tuple: (2D data array, label array, original 3D HSI array)
        """
        hsi_data = scipy.io.loadmat(self.hsi_path)
        X_3D = hsi_data['Botswana'].astype(np.float64)
        X_2D = X_3D.reshape(-1, X_3D.shape[-1])
        
        gt_data = scipy.io.loadmat(self.gt_path)
        y_true = gt_data['Botswana_gt'].reshape(-1).T.astype(np.int64)
        return X_2D, y_true, X_3D
    
    def _load_ksc(self):
        """Load KSC dataset
        
        Returns:
            tuple: (2D data array, label array, original 3D HSI array)
        """
        hsi_data = scipy.io.loadmat(self.hsi_path)
        X_3D = hsi_data['KSC'].astype(np.float64)
        X_2D = X_3D.reshape(-1, X_3D.shape[-1])
        
        gt_data = scipy.io.loadmat(self.gt_path)
        y_true = gt_data['KSC_gt'].reshape(-1).T
        return X_2D, y_true, X_3D
    
    def _load_salinas(self):
        """Load Salinas dataset
        
        Returns:
            tuple: (2D data array, label array, original 3D HSI array)
        """
        hsi_data = scipy.io.loadmat(self.hsi_path)
        X_3D = hsi_data['salinas_corrected'].astype(np.float64)

        X_2D = X_3D.reshape(-1, X_3D.shape[-1])
        
        gt_data = scipy.io.loadmat(self.gt_path)
        y_true = gt_data['salinas_gt'].reshape(-1).T
        return X_2D, y_true, X_3D

def load_noisy_data(dataset_name):
    """Load noisy HSI data.

    Args:
        dataset_name (str): Name of the dataset.
    Returns:
        tuple: (2D data array, label array, original 3D HSI array)
    """
    hsi_data_path = os.path.join(BASE_DATA_DIR, f'{dataset_name}', f'{dataset_name}_raw.npy')
    gt_data_path = os.path.join(BASE_DATA_DIR, f'{dataset_name}', f'{dataset_name}_gt.npy')
    X_3D = np.load(hsi_data_path)
    y_true = np.load(gt_data_path).reshape(-1).T
    X_2D = X_3D.reshape(-1, X_3D.shape[-1])
    return X_2D, y_true, X_3D
# Convenience function: quickly load HSI dataset
def load_hsi_data(dataset_name, preprocess=True):
    """Convenience function to quickly load hyperspectral image dataset
    
    Args:
        dataset_name (str): Name of the dataset to load
        preprocess (bool): Whether to apply preprocessing (default: True)
    
    Returns:
        tuple: (2D data array, label array, original 3D HSI array)
    """
    loader = HSIDataLoader(dataset_name, preprocess)
    return loader.load()
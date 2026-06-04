"""
Hyperspectral Image Noise Generation Module

This module provides functions for generating noisy hyperspectral images
by adding controlled Gaussian noise to specified spectral bands. The noise
is applied uniformly across randomly selected bands based on a given SNR.

Author: Kang Sun
"""

import numpy as np
from pathlib import Path


def generate_noisy_image(X_3D, noise_percent_list, save_path_parent, dataset_name, 
                         snr_db=15, gt_img=None, seed=None):
    """
    Generate noisy hyperspectral images with varying noise percentages and save to disk.
    
    This function creates multiple noisy versions of the input hyperspectral image
    by applying different noise percentages. Each noisy dataset and corresponding
    ground truth are saved as numpy files in organized directories.
    
    Args:
        X_3D (np.ndarray): Input 3D hyperspectral image array with shape (height, width, num_bands)
        noise_percent_list (list): List of noise percentages to apply (e.g., [30, 50, 70])
        save_path_parent (str or Path): Parent directory path for saving output files
        dataset_name (str): Name identifier for the dataset (used in filenames)
        snr_db (float, optional): Target signal-to-noise ratio in decibels. Defaults to 15.
        gt_img (np.ndarray, optional): Ground truth image array to save alongside noisy data.
            Defaults to None.
        seed (int, optional): Random seed for reproducible noise generation. Defaults to None.
    
    Returns:
        None: Files are saved directly to disk
        
    Notes:
        - Creates a subdirectory for each noise percentage
        - Skips generation if output files already exist (idempotent operation)
        - Saves three files per noise level: raw data, noise band indices, and ground truth
    """
    # Process each noise percentage configuration
    for noise_percent in noise_percent_list:
        # Create output directory structure
        save_path = Path.joinpath(Path(save_path_parent), f'{dataset_name}_{noise_percent}')
        Path.mkdir(save_path, exist_ok=True)
        
        # Define paths for output files
        full_save_path = Path.joinpath(save_path, f'{dataset_name}_{noise_percent}_raw.npy')
        
        # Generate and save noisy data if not already exists
        if not Path.exists(full_save_path):
            # Apply noise to randomly selected bands
            noisy_data, noise_band_index = _add_uniform_noise(X_3D, snr_db, noise_percent, seed)
            np.save(full_save_path, noisy_data)
            
            # Save indices of bands that were contaminated with noise
            noise_band_index_file_path = Path.joinpath(save_path, 
                                                      f'{dataset_name}_{noise_percent}_noise_band_index.npy')
            np.save(noise_band_index_file_path, noise_band_index)
        
        # Save ground truth image if provided and not already exists
        gt_file_path = Path.joinpath(save_path, f'{dataset_name}_{noise_percent}_gt.npy')
        if not Path.exists(gt_file_path) and gt_img is not None:
            np.save(gt_file_path, gt_img)


def _add_uniform_noise(data, snr_db, percent=50, seed=None):
    """
    Add zero-mean Gaussian white noise to a random subset of spectral bands.
    
    This private function applies additive Gaussian noise to a specified percentage
    of randomly selected spectral bands. The noise standard deviation is calculated
    based on the target SNR and the standard deviation of each band.
    
    Args:
        data (np.ndarray): Input 3D hyperspectral array with shape (height, width, num_bands)
        snr_db (float): Desired signal-to-noise ratio in decibels
        percent (float, optional): Percentage of bands to contaminate with noise. 
            Defaults to 50.
        seed (int, optional): Random seed for reproducible band selection and noise.
            Defaults to None.
    
    Returns:
        tuple:
            - noisy_data (np.ndarray): Noisy hyperspectral array with same shape as input
            - noise_band_index (np.ndarray): Array of band indices that received noise
    
    Notes:
        - Noise variance is computed per band based on the band's standard deviation
        - SNR formula: SNR_dB = 20 * log10(signal_std / noise_std)
        - Bands are selected without replacement
        - Original data is not modified (deep copy performed)
    """
    # Set random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate number of bands to contaminate based on percentage
    total_bands = data.shape[2]
    noisy_band_num = int(total_bands * percent / 100.0)
    
    # Randomly select which bands to add noise to (without replacement)
    noise_band_index = np.random.choice(total_bands, noisy_band_num, replace=False)
    
    # Create copy of original data to avoid modifying input
    noisy_data = data.copy().astype(np.float64)
    
    # Apply Gaussian noise to each selected band independently
    for band_ind in noise_band_index:
        # Calculate standard deviation of the clean signal in this band
        band_std = np.std(data[:, :, band_ind])
        
        # Convert SNR from dB to noise standard deviation
        # SNR_dB = 20 * log10(signal_std / noise_std)
        # => noise_std = signal_std / 10^(SNR_dB / 20)
        sigma_b = band_std / (10 ** (snr_db / 20.0))
        
        # Add zero-mean Gaussian noise to this band
        noisy_data[:, :, band_ind] += np.random.normal(0, sigma_b, data[:, :, band_ind].shape)
    
    return noisy_data, noise_band_index

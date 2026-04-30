"""
This is run after 02_benchmark_time.py, where you can save the reconstructions.
These are then loaded here to compare the quality of the reconstructions.

The comparison is made in respect to the FBP_CUDA reconstruction of 400 steps for the FL case

For the transmission case, the FBP_CUDA reconstruction of 800 steps is used.
"""

import numpy as np
import tomopy as tom
import os
import sys
from pathlib import Path
import gc
from tqdm import tqdm
from time import perf_counter
import matplotlib.pyplot as plt
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

project_root = Path(__file__).resolve().parents[1]
if not (project_root / "utils").exists():
    raise Exception(f"You have to keep the original repository structure, current project root: {project_root}")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils import utils_opt as u


UNDERSAMPLE = 10

def load_recon(path):
    return np.load(path)


def compare_reconstructions(recon, ground_truth):
    mse = mean_squared_error(
        ground_truth[::UNDERSAMPLE],
        recon[::UNDERSAMPLE],
    )

    # For SSIM, compare slice-wise and average
    ssim_list, psnr_list = [], []
    psnr = peak_signal_noise_ratio(
        ground_truth, recon,
        data_range=ground_truth.max() - ground_truth.min(),
    )
    psnr_list.append(psnr)
    ssim = structural_similarity(
        ground_truth, recon,
        data_range=ground_truth.max() - ground_truth.min(),
    )
    ssim_list.append(ssim)
    out = {
        'MSE': mse,
        'PSNR': psnr_list,
        'SSIM': ssim_list,
        }
    return out


def compare_reconstructions_undersample_gt(recon, ground_truth, undersample):
    gt = ground_truth[::undersample].copy()
    mse = mean_squared_error(
        gt,
        recon,
    )

    assert gt.shape == recon.shape, f"Ground truth shape {gt.shape} does not match reconstruction shape {recon.shape}"
    # For SSIM, compare slice-wise and average
    ssim_list, psnr_list = [], []
    psnr = peak_signal_noise_ratio(
        gt, recon,
        data_range=gt.max() - gt.min(),
    )
    psnr_list.append(psnr)
    ssim = structural_similarity(
        gt, recon,
        data_range=gt.max() - gt.min(),
    )
    ssim_list.append(ssim)
    out = {
        'MSE': mse,
        'PSNR': psnr_list,
        'SSIM': ssim_list,
        }
    return out

# This works for recons which were not undersampled
# Paths to your reconstructions and ground truth
recon_paths = [
    project_root / 'benchmarks/results/0801_fl_lp590_25_tomopy_fbp_gpu_recon.npy',
    project_root / 'benchmarks/results/0801_fl_lp590_25_tomopari_fbp_gpu_recon.npy',
    project_root / 'benchmarks/results/0801_fl_lp590_50_tomopy_fbp_gpu_recon.npy',
    project_root / 'benchmarks/results/0801_fl_lp590_50_tomopari_fbp_gpu_recon.npy',
]
GT_path = project_root / 'benchmarks/results/0801_fl_lp590_400_tomopy_fbp_gpu_recon.npy'  # or a true ground truth recon

# Load ground truth reconstruction (should be a full recon, not raw data)
GT = load_recon(GT_path)
print(f"Ground truth shape: {GT.shape}")


METRICS = {}
for path in recon_paths:
    recon = load_recon(path)
    print(f"Loaded reconstruction from {path} with shape: {recon.shape}")
    assert recon.shape == GT.shape, f"Reconstruction shape {recon.shape} does not match ground truth shape {GT.shape}"
    metrics = compare_reconstructions(recon, GT)
    print(metrics)
    METRICS[os.path.basename(path)] = metrics
    print('##################################')


## Undersampled paths ##
########################
recon_paths = [
    project_root / 'benchmarks/results/0801_fl_lp590_25_tomopari_fbp_cpu_recon.npy',
    project_root / 'benchmarks/results/0801_fl_lp590_25_tomopari_tomodl_gpu_recon.npy',
    project_root / 'benchmarks/results/0801_fl_lp590_25_tomopari_tomodl_cpu_recon.npy',
    project_root / 'benchmarks/results/0801_fl_lp590_25_tomopy_sart_gpu_recon.npy',
    project_root / 'benchmarks/results/0801_fl_lp590_50_tomopari_fbp_cpu_recon.npy',
    project_root / 'benchmarks/results/0801_fl_lp590_50_tomopari_tomodl_gpu_recon.npy',
    project_root / 'benchmarks/results/0801_fl_lp590_50_tomopari_tomodl_cpu_recon.npy',
    project_root / 'benchmarks/results/0801_fl_lp590_50_tomopy_sart_gpu_recon.npy',
]

for path in recon_paths:
    recon = load_recon(path)
    print(f"Loaded undersampled reconstruction from {path} with shape: {recon.shape}")
    metrics = compare_reconstructions_undersample_gt(recon, GT, 200)
    print(metrics)
    METRICS[os.path.basename(path)] = metrics
    print('##################################')


#save METRICS
np.save(project_root / 'benchmarks/results/metrics_FL.npy', METRICS)

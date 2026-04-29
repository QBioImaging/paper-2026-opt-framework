"""
This is run after 02_benchmark_time.py, where you can save the reconstructions.
These are then loaded here to compare the quality of the reconstructions.

The comparison is made in respect to the FBP_CUDA reconstruction of 400 steps for the FL case

For the transmission case, the FBP_CUDA reconstruction of 800 steps is used.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path.cwd().resolve().parent
UNDERSAMPLE = 10
metrics = np.load(project_root / 'benchmarks/results/metrics_FL.npy', allow_pickle=True).item()

fig, ax = plt.subplots()
for key, value in metrics.items():
    for metric_name, metric_values in value.items():
        if metric_name == 'PSNR':
            plt.barh(key, np.mean(metric_values), xerr=np.std(metric_values), label=metric_name)
plt.xlabel("PSNR (dB), mean per slice")
plt.title("Quality over slices, Fluorescence")
plt.tight_layout()
plt.savefig(project_root / 'benchmarks/results/psnr_fl.png')
plt.show()


fig, ax = plt.subplots()
for key, value in metrics.items():
    for metric_name, metric_values in value.items():
        if metric_name == 'SSIM':
            plt.barh(key, np.mean(metric_values), xerr=np.std(metric_values), label=metric_name)
plt.xlabel("SSIM, mean per slice")
plt.title("Quality over slices, Fluorescence")
plt.tight_layout()
plt.savefig(project_root / 'benchmarks/results/ssim_fl.png')
plt.show()
"""
This is run after 02_benchmark_time.py, where you can save the reconstructions.
These are then loaded here to compare the quality of the reconstructions.

The comparison is made in respect to the FBP_CUDA reconstruction of 400 steps for the FL case

For the transmission case, the FBP_CUDA reconstruction of 800 steps is used.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
UNDERSAMPLE = 10
metrics = np.load(project_root / 'benchmarks/results/metrics_FL.npy', allow_pickle=True).item()

_ACRONYMS = {'fbp', 'cpu', 'gpu', 'sart', 'cuda', 'art', 'tv', 'mlem', 'gridrec'}
_STEP_COLORS = {'25': 'tab:blue', '50': 'tab:orange'}
_DEFAULT_COLOR = 'tab:gray'


def _format_label(raw: str) -> str:
    parts = raw.replace('-', ' ').split('_')
    return ' '.join(p.upper() if p.lower() in _ACRONYMS else p for p in parts)


def _bar_color(raw: str) -> str:
    step = raw.split('_')[0]
    return _STEP_COLORS.get(step, _DEFAULT_COLOR)

psnr_data = {
    key.replace('0801_fl_lp590_', '').replace('_recon.npy', ''): (np.mean(v['PSNR']), np.std(v['PSNR']))
    for key, v in metrics.items() if 'PSNR' in v
}
psnr_sorted = sorted(psnr_data.items(), key=lambda x: x[1][0], reverse=False)
fig, ax = plt.subplots()
for raw_label, (mean, std) in psnr_sorted:
    ax.barh(_format_label(raw_label), mean, xerr=std, color=_bar_color(raw_label))
ax.set_xlabel("PSNR (dB), mean over slices")
ax.set_title("Quality over slices, Fluorescence")
plt.tight_layout()
plt.savefig(project_root / 'benchmarks/results/psnr_fl.png')
plt.show()


ssim_data = {
    key.replace('0801_fl_lp590_', '').replace('_recon.npy', ''): (np.mean(v['SSIM']), np.std(v['SSIM']))
    for key, v in metrics.items() if 'SSIM' in v
}
ssim_sorted = sorted(ssim_data.items(), key=lambda x: x[1][0], reverse=False)
fig, ax = plt.subplots()
for raw_label, (mean, std) in ssim_sorted:
    ax.barh(_format_label(raw_label), mean, xerr=std, color=_bar_color(raw_label))
ax.set_xlabel("SSIM, mean over slices")
ax.set_title("Quality over slices, Fluorescence")
plt.tight_layout()
plt.savefig(project_root / 'benchmarks/results/ssim_fl.png')
plt.show()
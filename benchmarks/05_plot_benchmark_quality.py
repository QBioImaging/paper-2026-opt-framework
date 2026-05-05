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

_ACRONYMS = {'fbp', 'cpu', 'gpu', 'sart', 'cuda', 'art', 'tv', 'mlem', 'gridrec', 'tomodl', 'tr', 'fl'}
_STEP_COLORS = {'25': 'tab:blue', '50': 'tab:orange', '400': 'tab:green', '800': 'tab:red'}
_DEFAULT_COLOR = 'tab:gray'


def _format_label(raw: str) -> str:
    # key format after prefix strip: {steps}_{method}
    tokens = raw.replace('-', '_').split('_')
    method_parts = tokens[1:]  # skip step count
    method_parts = [p for p in method_parts if p.lower() != 'stride' and not p.isdigit()]
    return ' '.join(p.upper() if p.lower() in _ACRONYMS else p for p in method_parts if p)


def _bar_color(raw: str) -> str:
    step = raw.split('_')[0]
    return _STEP_COLORS.get(step, _DEFAULT_COLOR)


def _add_legend(ax, sorted_items: list) -> None:
    present_steps = sorted({raw.split('_')[0] for raw, _ in sorted_items if raw.split('_')[0] in _STEP_COLORS})
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=_STEP_COLORS[step], label=f"{step} steps")
        for step in present_steps
    ]
    if legend_handles:
        ax.legend(handles=legend_handles)

psnr_data = {
    key.replace('0801_fl_lp590_', '').replace('_recon.npy', ''): (np.mean(v['PSNR']), np.std(v['PSNR']))
    for key, v in metrics.items() if 'PSNR' in v
}
psnr_sorted = sorted(psnr_data.items(), key=lambda x: x[1][0], reverse=False)
fig, ax = plt.subplots()
for i, (raw_label, (mean, std)) in enumerate(psnr_sorted):
    ax.barh(i, mean, xerr=std, color=_bar_color(raw_label))
ax.set_yticks(range(len(psnr_sorted)))
ax.set_yticklabels([_format_label(raw_label) for raw_label, _ in psnr_sorted])
_add_legend(ax, psnr_sorted)
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
for i, (raw_label, (mean, std)) in enumerate(ssim_sorted):
    ax.barh(i, mean, xerr=std, color=_bar_color(raw_label))
ax.set_yticks(range(len(ssim_sorted)))
ax.set_yticklabels([_format_label(raw_label) for raw_label, _ in ssim_sorted])
_add_legend(ax, ssim_sorted)
ax.set_xlabel("SSIM, mean over slices")
ax.set_title("Quality over slices, Fluorescence")
plt.tight_layout()
plt.savefig(project_root / 'benchmarks/results/ssim_fl.png')
plt.show()
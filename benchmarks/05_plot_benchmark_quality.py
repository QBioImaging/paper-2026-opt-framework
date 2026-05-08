"""
This is run after 04_benchmark_quality.py, where you can save the reconstructions.
These are then loaded here to compare the quality of the reconstructions.

The comparison is made in respect to the FBP_CUDA reconstruction of 400 steps for
the FL case

For the transmission case, the FBP_CUDA reconstruction of 800 steps is used.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

project_root = Path(__file__).resolve().parents[1]
metrics_fl = np.load(project_root / 'benchmarks/results/metrics_FL.npy', allow_pickle=True).item()
metrics_tr = np.load(project_root / 'benchmarks/results/metrics_TR.npy', allow_pickle=True).item()
PLOT_MODES = ('raw', 'norm', 'flex')

_ACRONYMS = {'fbp', 'cpu', 'gpu', 'sart', 'cuda', 'art', 'tv', 'mlem', 'gridrec', 'tomodl', 'tr', 'fl'}
_STEP_COLORS = {
    '400': '#D55E00', # 1x  — Vermillion
    '50':  '#009E73', # 8x  — Bluish Green
    '25':  '#E69F00', # 16x — Orange
}
_STEP_LABELS = {'400': '1x', '50': '8x', '25': '16x'}
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


def _add_legend(ax, sorted_items: list, pos=(1.05, 0.5)) -> None:
    present_steps = {raw.split('_')[0] for raw, _ in sorted_items if raw.split('_')[0] in _STEP_COLORS}
    # Use _STEP_COLORS key order (1x → 8x → 16x)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=_STEP_COLORS[step], label=_STEP_LABELS[step])
        for step in _STEP_COLORS
        if step in present_steps
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, bbox_to_anchor=pos)

_FILENAME_PATTERN = re.compile(
    r"(?P<prefix>.+)_(?P<modality>[a-z]+)_lp\d+_(?P<steps>\d+)_(?P<method>.+)-(?P<undersample>\d+)$"
)


def _key_to_label(filename: str, modality: str) -> str:
    match = _FILENAME_PATTERN.match(filename)
    if match and match.group('modality') == modality:
        return f"{match.group('steps')}_{match.group('method')}"
    return filename.replace('_recon.npy', '')


def _metric_series(metrics: dict, modality: str, mode: str, metric_name: str) -> dict[str, tuple[float, float]]:
    series: dict[str, tuple[float, float]] = {}
    for filename, record in metrics.items():
        if mode not in record:
            continue
        metric_values = np.asarray(record[mode].get(metric_name, []), dtype=float)
        if metric_name == 'PSNR':
            metric_values = metric_values[np.isfinite(metric_values)]
        if metric_values.size == 0:
            continue
        label = _key_to_label(filename, modality)
        series[label] = (float(np.nanmean(metric_values)), float(np.nanstd(metric_values)))
    return series


def _plot_row(ax_left, ax_right, metrics: dict, modality: str, mode: str, show_xlabel: bool) -> None:
    psnr_data = _metric_series(metrics, modality, mode, 'PSNR')
    ssim_data = _metric_series(metrics, modality, mode, 'SSIM')

    psnr_sorted = sorted(psnr_data.items(), key=lambda x: x[1][0], reverse=False)
    ssim_lookup = dict(ssim_data.items())
    ordered_labels = [label for label, _ in psnr_sorted if label in ssim_lookup]

    if not ordered_labels:
        ax_left.text(0.5, 0.5, f'No data for {mode}', ha='center', va='center', transform=ax_left.transAxes)
        ax_right.text(0.5, 0.5, f'No data for {mode}', ha='center', va='center', transform=ax_right.transAxes)
        return

    for i, label in enumerate(ordered_labels):
        psnr_mean, psnr_std = psnr_data[label]
        ax_left.barh(i, psnr_mean, xerr=psnr_std, color=_bar_color(label))
        ax_left.set_xlim(0, 48)

    for i, label in enumerate(ordered_labels):
        ssim_mean, ssim_std = ssim_lookup[label]
        ax_right.barh(i, ssim_mean, xerr=ssim_std, color=_bar_color(label))
        ax_right.set_xlim(0, 1.05)

    ax_left.set_yticks(range(len(ordered_labels)))
    ax_left.set_yticklabels([_format_label(label) for label in ordered_labels])
    ax_right.set_yticks(range(len(ordered_labels)))
    ax_right.set_yticklabels([_format_label(label) for label in ordered_labels])
    ax_right.tick_params(axis='y', labelleft=False)

    if show_xlabel:
        ax_left.set_xlabel('PSNR (dB), mean over slices')
        ax_right.set_xlabel('SSIM, mean over slices')

    legend_items = [(label, (0.0, 0.0)) for label in ordered_labels]
    if modality == 'fl':
        _add_legend(ax_left, legend_items, pos=(-0.08, -0.02))


def _plot_mode(mode: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 12), nrows=2, ncols=2, sharey='row')

    _plot_row(ax[0, 0], ax[0, 1], metrics_fl, 'fl', mode, show_xlabel=False)
    _plot_row(ax[1, 0], ax[1, 1], metrics_tr, 'tr', mode, show_xlabel=True)

    fig.text(0.55, 0.97, 'Fluorescence', ha='center', va='top', fontsize=14)
    fig.text(0.55, 0.48, 'Transmission', ha='center', va='top', fontsize=14)
    plt.tight_layout(h_pad=1.5, rect=[0, 0, 1, 0.96])
    plt.savefig(project_root / f'fig_output/quality_all_{mode}.png')
    plt.show()


for mode in PLOT_MODES:
    _plot_mode(mode)
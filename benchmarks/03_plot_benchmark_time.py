import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

project_root = Path(__file__).resolve().parents[1]

_ACRONYMS = {'fbp', 'cpu', 'gpu', 'sart', 'cuda', 'art', 'tv', 'mlem', 'gridrec', 'tomodl', 'tr', 'fl'}
_STEP_COLORS = {
    '400': '#D55E00', # 1x  — Vermillion
    '50':  '#009E73', # 8x  — Bluish Green
    '25':  '#E69F00', # 16x — Orange
}
_STEP_LABELS = {'400': '1x', '50': '8x', '25': '16x'}
_DEFAULT_COLOR = 'tab:gray'


def _format_label(raw: str) -> str:
    # key format: {modality}_{steps}_{method_key_suffix}
    tokens = raw.replace('-', '_').split('_')
    method_parts = tokens[2:]  # skip modality and step
    method_parts = [p for p in method_parts if p.lower() != 'stride' and not p.isdigit()]
    return ' '.join(p.upper() if p.lower() in _ACRONYMS else p for p in method_parts if p)


def _bar_color(raw: str) -> str:
    # step is the second token: {modality}_{steps}_...
    tokens = raw.split('_')
    step = tokens[1] if len(tokens) > 1 else ''
    return _STEP_COLORS.get(step, _DEFAULT_COLOR)


def _plot_times(bdict: dict, title: str, save_path: Path) -> None:
    avg = {k: np.mean(v) for k, v in bdict.items()}
    std = {k: np.std(v) for k, v in bdict.items()}
    print(f"\n{title}")
    for k, v in avg.items():
        print(f"  {k}: {v:.2f} +- {std[k]:.2f} seconds")

    sorted_items = sorted(avg.items(), key=lambda x: x[1], reverse=True)
    _, ax = plt.subplots()
    for i, (raw_key, mean) in enumerate(sorted_items):
        ax.barh(i, mean, xerr=std[raw_key], color=_bar_color(raw_key))
    ax.set_yticks(range(len(sorted_items)))
    ax.set_yticklabels([_format_label(raw_key) for raw_key, _ in sorted_items])

    present_steps = sorted({raw.split('_')[1] for raw, _ in sorted_items if raw.split('_')[1] in _STEP_COLORS})
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=_STEP_COLORS[step], label=f"{_STEP_LABELS[step]}")
        for step in _STEP_COLORS
        if step in present_steps
    ]
    if legend_handles:
        ax.legend(handles=legend_handles)

    ax.set_xscale("log")
    ax.set_xlabel("Time (s)")
    # ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


# Fluorescence
with open(project_root / 'benchmarks/results/benchmarks_fl.npy', 'rb') as f:
    BDICT = np.load(f, allow_pickle=True).item()
_plot_times(BDICT, "Average Reconstruction Times, Fluorescence",
            project_root / 'fig_output/average_reconstruction_times_fl.png')

# Transmission
with open(project_root / 'benchmarks/results/benchmarks_tr.npy', 'rb') as f:
    BDICT = np.load(f, allow_pickle=True).item()
_plot_times(BDICT, "Average Reconstruction Times, Transmission",
            project_root / 'fig_output/average_reconstruction_times_tr.png')

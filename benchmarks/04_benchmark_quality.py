"""Compute reconstruction quality metrics from discovered benchmark outputs."""

from dataclasses import dataclass
from pathlib import Path
import re
import sys

import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity


project_root = Path(__file__).resolve().parents[1]
if not (project_root / "utils").exists():
    raise RuntimeError(
        "You have to keep the original repository structure, "
        f"current project root: {project_root}"
    )
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


RESULTS_DIR = project_root / "benchmarks" / "results"
OUTPUT_METRICS_TEMPLATE = "metrics_{modality}.npy"
FILENAME_PATTERN = re.compile(
    r"(?P<prefix>.+)_(?P<modality>[a-z]+)_lp\d+_(?P<steps>\d+)_(?P<method>.+)_recon\.npy$"
)


@dataclass(frozen=True)
class ReconstructionRecord:
    path: Path
    params_path: Path
    modality: str
    steps: int
    method: str
    undersample: int


def load_recon(path: Path) -> np.ndarray:
    return np.load(path)


def load_recon_params(params_path: Path) -> dict:
    params = np.load(params_path, allow_pickle=True)
    if isinstance(params, np.ndarray) and params.shape == ():
        return params.item()
    if isinstance(params, dict):
        return params
    raise ValueError(f"Unexpected params format in {params_path}")


def compare_reconstruction(recon: np.ndarray, ground_truth: np.ndarray, undersample: int) -> dict:
    aligned_ground_truth = ground_truth[::undersample].copy() if undersample > 1 else ground_truth
    if aligned_ground_truth.shape != recon.shape:
        raise ValueError(
            f"Ground truth shape {aligned_ground_truth.shape} does not match "
            f"reconstruction shape {recon.shape} after undersample={undersample}"
        )

    mse = mean_squared_error(aligned_ground_truth, recon)
    data_range = aligned_ground_truth.max() - aligned_ground_truth.min()
    psnr = peak_signal_noise_ratio(aligned_ground_truth, recon, data_range=data_range)
    ssim = structural_similarity(aligned_ground_truth, recon, data_range=data_range)
    return {
        "MSE": mse,
        "PSNR": [psnr],
        "SSIM": [ssim],
    }


def parse_record(recon_path: Path) -> ReconstructionRecord | None:
    match = FILENAME_PATTERN.match(recon_path.name)
    if match is None:
        return None

    params_path = recon_path.with_name(recon_path.name.replace(".npy", "_params.npy"))
    if not params_path.exists():
        print(f"Skipping {recon_path.name}: missing params file {params_path.name}")
        return None

    params = load_recon_params(params_path)
    undersample = int(params.get("undersample", 1))
    return ReconstructionRecord(
        path=recon_path,
        params_path=params_path,
        modality=match.group("modality"),
        steps=int(match.group("steps")),
        method=match.group("method"),
        undersample=undersample,
    )


def discover_reconstructions(results_dir: Path) -> list[ReconstructionRecord]:
    records: list[ReconstructionRecord] = []
    for recon_path in sorted(results_dir.glob("*_recon.npy")):
        record = parse_record(recon_path)
        if record is not None:
            records.append(record)
    return records


def select_ground_truth(records: list[ReconstructionRecord], modality: str) -> ReconstructionRecord:
    candidates = [
        record
        for record in records
        if record.modality == modality
        and record.method == "tomopy_fbp_gpu"
        and record.undersample == 1
    ]
    if not candidates:
        raise RuntimeError(
            f"No tomopy_fbp_gpu reconstruction with undersample=1 found for modality '{modality}'"
        )
    return max(candidates, key=lambda record: record.steps)


def build_metrics(records: list[ReconstructionRecord], modality: str) -> dict[str, dict]:
    ground_truth_record = select_ground_truth(records, modality)
    ground_truth = load_recon(ground_truth_record.path)
    print(f"Ground truth: {ground_truth_record.path.name} shape={ground_truth.shape}")

    metrics: dict[str, dict] = {}
    for record in records:
        if record.modality != modality:
            continue
        if record.path == ground_truth_record.path:
            print(f"Skipping {record.path.name}: this is the ground truth")
            continue

        recon = load_recon(record.path)
        print(
            f"Loaded {record.path.name} with shape={recon.shape} "
            f"undersample={record.undersample}"
        )
        record_metrics = compare_reconstruction(recon, ground_truth, record.undersample)
        print(record_metrics)
        metrics[record.path.name] = record_metrics
        print("##################################")

    return metrics


def main() -> None:
    records = discover_reconstructions(RESULTS_DIR)
    if not records:
        raise RuntimeError(f"No reconstruction files found in {RESULTS_DIR}")

    modalities = sorted({record.modality for record in records})
    if not modalities:
        raise RuntimeError(f"No parseable reconstruction files found in {RESULTS_DIR}")

    for modality in modalities:
        modality_metrics = build_metrics(records, modality)
        output_path = RESULTS_DIR / OUTPUT_METRICS_TEMPLATE.format(modality=modality.upper())
        np.save(output_path, modality_metrics)
        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
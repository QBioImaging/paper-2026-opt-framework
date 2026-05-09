from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import tomopy as tom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if not (PROJECT_ROOT / "utils").exists():
    raise RuntimeError(
        "You have to keep the original repository structure, "
        f"current project root: {PROJECT_ROOT}"
    )
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import utils_opt as u


RESIZE = 256
CENTER = 779
CIRC_MASK = 1.0
RUN_BENCHMARKS = True
REPEATS = 3
PLOT = True
STEPS_TO_RUN = (25, 50, 400)
MODALITY_KEY = "tr"
MODALITY_LABEL = "Transmission"

RESULTS_DIR = PROJECT_ROOT / "benchmarks" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class MethodSpec:
    name: str
    caller: str
    key_suffix: str
    params: dict
    save_name: str | None


METHODS_BY_STEP: dict[int, list[MethodSpec]] = {
    25: [
        MethodSpec(
            name="FBP GPU",
            caller="tomopy",
            key_suffix="tomopy_fbp_gpu_stride-1",
            params={
                "undersample": 1,
                "algorithm": tom.astra,
                "options": {"proj_type": "cuda", "method": "FBP_CUDA"},
                "ncore": 1,
                "plot_title": "FBP GPU TR 25 Reconstruction",
            },
            save_name="0801_tr_lp590_25_tomopy_fbp_gpu_recon.npy",
        ),
        MethodSpec(
            name="FBP GPU",
            caller="tomopari",
            key_suffix="tomopari_fbp_gpu_stride-1",
            params={
                "method": u.Rec_Modes.FBP_GPU.value,
                "filter": "ramp", # can be  "ramp", "shepp-logan" or "cosine" or "hamming" or "hann"
                "undersample": 1,
                "order_mode": 0,
                "invert_color": False,
                "batch_process": 32,
                "is_half_rotation": False,
                "clip_to_circle": True,
                "plot_title": "FBP GPU TR 25 Reconstruction",
            },
            save_name="0801_tr_lp590_25_tomopari_fbp_gpu_recon.npy",
        ),
        MethodSpec(
            name="FBP CPU",
            caller="tomopari",
            key_suffix="tomopari_fbp_cpu_stride-100",
            params={
                "method": u.Rec_Modes.FBP_CPU.value,
                "filter": "ramp", # can be  "ramp", "shepp-logan" or "cosine" or "hamming" or "hann"
                "undersample": 100,
                "order_mode": 0,
                "invert_color": False,
                "batch_process": 1,
                "is_half_rotation": False,
                "clip_to_circle": True,
                "plot_title": "FBP CPU TR 25 Reconstruction",
            },
            save_name="0801_tr_lp590_25_tomopari_fbp_cpu_recon.npy",
        ),
        MethodSpec(
            name="TOMODL GPU",
            caller="tomopari",
            key_suffix="tomopari_tomodl_gpu_stride-100",
            params={
                "method": u.Rec_Modes.TOMODL_GPU.value,
                "filter": "ramp", # can be  "ramp", "shepp-logan" or "cosine" or "hamming" or "hann"
                "undersample": 100,
                "order_mode": 0,
                "invert_color": False,
                "batch_process": 12,
                "is_half_rotation": False,
                "clip_to_circle": True,
                "plot_title": "TOMODL GPU TR 25 Reconstruction",
            },
            save_name="0801_tr_lp590_25_tomopari_tomodl_gpu_recon.npy",
        ),
        MethodSpec(
            name="TOMODL CPU",
            caller="tomopari",
            key_suffix="tomopari_tomodl_cpu_stride-100",
            params={
                "method": u.Rec_Modes.TOMODL_CPU.value,
                "filter": "ramp", # can be  "ramp", "shepp-logan" or "cosine" or "hamming" or "hann"
                "undersample": 100,
                "order_mode": 0,
                "batch_process": 1,
                "invert_color": False,
                "is_half_rotation": False,
                "clip_to_circle": True,
                "plot_title": "TOMODL CPU TR 25 Reconstruction",
            },
            save_name="0801_tr_lp590_25_tomopari_tomodl_cpu_recon.npy",
        ),
        MethodSpec(
            name="SART GPU",
            caller="tomopy",
            key_suffix="tomopy_sart_gpu_stride-100",
            params={
                "undersample": 100,
                "algorithm": tom.astra,
                "options": {
                    "method": "SART",
                    "num_iter": 200,
                    "proj_type": "linear",
                    "extra_options": {"MinConstraint": 0},
                },
                "plot_title": "SART GPU TR 25 Reconstruction",
            },
            save_name="0801_tr_lp590_25_tomopy_sart_gpu_recon.npy",
        ),
    ],
    50: [
        MethodSpec(
            name="FBP GPU",
            caller="tomopy",
            key_suffix="tomopy_fbp_gpu_stride-1",
            params={
                "undersample": 1,
                "algorithm": tom.astra,
                "options": {"proj_type": "cuda", "method": "FBP_CUDA"},
                "ncore": 1,
                "plot_title": "FBP GPU TR 50 Reconstruction",
            },
            save_name="0801_tr_lp590_50_tomopy_fbp_gpu_recon.npy",
        ),
        MethodSpec(
            name="FBP GPU",
            caller="tomopari",
            key_suffix="tomopari_fbp_gpu_stride-1",
            params={
                "method": u.Rec_Modes.FBP_GPU.value,
                "filter": "ramp", # can be  "ramp", "shepp-logan" or "cosine" or "hamming" or "hann"
                "undersample": 1,
                "order_mode": 0,
                "invert_color": False,
                "batch_process": 32,
                "is_half_rotation": False,
                "clip_to_circle": True,
                "plot_title": "FBP GPU TR 50 Reconstruction",
            },
            save_name="0801_tr_lp590_50_tomopari_fbp_gpu_recon.npy",
        ),
        MethodSpec(
            name="FBP CPU",
            caller="tomopari",
            key_suffix="tomopari_fbp_cpu_stride-100",
            params={
                "method": u.Rec_Modes.FBP_CPU.value,
                "filter": "ramp", # can be  "ramp", "shepp-logan" or "cosine" or "hamming" or "hann"
                "undersample": 100,
                "order_mode": 0,
                "invert_color": False,
                "batch_process": 1,
                "is_half_rotation": False,
                "clip_to_circle": True,
                "plot_title": "FBP CPU TR 50 Reconstruction",
            },
            save_name="0801_tr_lp590_50_tomopari_fbp_cpu_recon.npy",
        ),
        MethodSpec(
            name="TOMODL GPU",
            caller="tomopari",
            key_suffix="tomopari_tomodl_gpu_stride-100",
            params={
                "method": u.Rec_Modes.TOMODL_GPU.value,
                "filter": "ramp", # can be  "ramp", "shepp-logan" or "cosine" or "hamming" or "hann"
                "undersample": 100,
                "order_mode": 0,
                "invert_color": False,
                "batch_process": 12,
                "is_half_rotation": False,
                "clip_to_circle": True,
                "plot_title": "TOMODL GPU TR 50 Reconstruction",
            },
            save_name="0801_tr_lp590_50_tomopari_tomodl_gpu_recon.npy",
        ),
        MethodSpec(
            name="TOMODL CPU",
            caller="tomopari",
            key_suffix="tomopari_tomodl_cpu_stride-100",
            params={
                "method": u.Rec_Modes.TOMODL_CPU.value,
                "filter": "ramp", # can be  "ramp", "shepp-logan" or "cosine" or "hamming" or "hann"
                "undersample": 100,
                "order_mode": 0,
                "invert_color": False,
                "batch_process": 1,
                "is_half_rotation": False,
                "clip_to_circle": True,
                "plot_title": "TOMODL CPU TR 50 Reconstruction",
            },
            save_name="0801_tr_lp590_50_tomopari_tomodl_cpu_recon.npy",
        ),
        MethodSpec(
            name="SART GPU",
            caller="tomopy",
            key_suffix="tomopy_sart_gpu_stride-100",
            params={
                "undersample": 100,
                "algorithm": tom.astra,
                "options": {
                    "method": "SART",
                    "num_iter": 200,
                    "proj_type": "linear",
                    "extra_options": {"MinConstraint": 0},
                },
                "plot_title": "SART GPU TR 50 Reconstruction",
            },
            save_name="0801_tr_lp590_50_tomopy_sart_gpu_recon.npy",
        ),
    ],
    400: [
        MethodSpec(
            name="FBP GPU",
            caller="tomopy",
            key_suffix="tomopy_fbp_gpu_stride-1",
            params={
                "undersample": 1,
                "algorithm": tom.astra,
                "options": {"proj_type": "cuda", "method": "FBP_CUDA"},
                "ncore": 1,
                "plot_title": "FBP GPU TR 400 Reconstruction",
            },
            save_name="0801_tr_lp590_400_tomopy_fbp_gpu_recon.npy",
        ),
        MethodSpec(
            name="SART GPU",
            caller="tomopy",
            key_suffix="tomopy_sart_gpu_stride-100",
            params={
                "undersample": 100,
                "algorithm": tom.astra,
                "options": {
                    "method": "SART",
                    "num_iter": 200,
                    "proj_type": "linear",
                    "extra_options": {"MinConstraint": 0},
                },
                "plot_title": "SART GPU TR 400 Reconstruction",
            },
            save_name="0801_tr_lp590_400_tomopy_sart_gpu_recon.npy",
        ),
    ],
}


def _dataset_path(step: int) -> Path:
    return PROJECT_ROOT / f"processed_data/0801_{MODALITY_KEY}_lp590_{step}_clean.npy"


def _base_params(thetas: np.ndarray, save_path: str | None, plot_title: str) -> dict:
    return {
        "resize_row": RESIZE,
        "thetas": thetas,
        "center": CENTER,
        "circ_mask": CIRC_MASK,
        "save_path": save_path,
        "plot": PLOT,
        "plot_title": plot_title,
    }


def _run_method(
    step: int,
    data: np.ndarray,
    thetas: np.ndarray,
    method: MethodSpec,
    benchmark_dict: dict[str, list[float]],
) -> None:
    print("\n ##################### \n")
    print(f"Running {method.name} for {MODALITY_KEY.upper()} {step} steps")

    save_path = None
    if method.save_name is not None:
        save_path = str(RESULTS_DIR / method.save_name)

    params = _base_params(thetas, save_path, method.params["plot_title"])
    params.update(method.params)

    if method.caller == "tomopy":
        runner = u.run_reconstruction
    elif method.caller == "tomopari":
        runner = u.reconstruct_tomopari
    else:
        raise ValueError(f"Unknown method caller: {method.caller}")

    _, first_time = runner(data, params)
    if not RUN_BENCHMARKS:
        return

    key = f"{MODALITY_KEY}_{step}_{method.key_suffix}"
    times = [first_time]

    for _ in range(REPEATS - 1):
        repeat_params = dict(params)
        repeat_params["save_path"] = None
        _, elapsed = runner(data, repeat_params)
        times.append(elapsed)

    benchmark_dict[key] = times


def run_step(step: int, benchmark_dict: dict[str, list[float]]) -> None:
    print("######################################################")
    print(f"###### Processing {step} step {MODALITY_LABEL} data... ######")

    data, thetas = u.load_data(_dataset_path(step))
    for method in METHODS_BY_STEP[step]:
        _run_method(step, data, thetas, method, benchmark_dict)


def main() -> None:
    benchmark_dict: dict[str, list[float]] = {}

    for step in STEPS_TO_RUN:
        if step not in METHODS_BY_STEP:
            raise ValueError(f"Unsupported step value: {step}")
        run_step(step, benchmark_dict)

    if RUN_BENCHMARKS:
        out_path = RESULTS_DIR / f"benchmarks_{MODALITY_KEY}.npy"
        with open(out_path, "wb") as f:
            np.save(f, benchmark_dict)
        print(f"Saved benchmark timings to {out_path}")


if __name__ == "__main__":
    main()

import numpy as np
import tomopy as tom
import sys
from pathlib import Path

project_root = Path.cwd().resolve().parent
if not (project_root / "utils").exists():
    raise Exception(f"You have to keep the original repository structure, current project root: {project_root}")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils import utils_opt as u

RESIZE = 256  # resize data before reconstruction
CENTER = 779
CIRC_MASK = 1.0
BDICT = {}
RUN_BENCHMARKS = True


###########################
## 25 step Transmission ##
###########################
print('####################################################')
print("##### Processing 25 step Transmission data... ######")
# 1. Load data from save path in raw2clean.py
data, thetas = u.load_data(project_root / 'processed_data/0801_tr_lp590_25_clean.npy')

## FBP CUDA from astra toolbox ##
params = {
    'undersample': 1,
    'resize_row': RESIZE,
    'thetas': thetas,
    'center': CENTER,
    'algorithm': tom.astra,
    'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA'},
    'ncore': 1,
    'circ_mask': CIRC_MASK,
    'save_path': str(project_root / 'benchmarks/results/0801_tr_lp590_25_recon.npy'),
    'plot': True,
    'plot_title': 'FBP CUDA TR 25 Reconstruction'
}

recon_time = u.run_reconstruction(data, params)

if RUN_BENCHMARKS:
    BDICT[f'tr_25_cuda_tomopy_fbp_stride-{params["undersample"]}'] = [recon_time]
    params['save_path'] = None
    for i in range(2):
        recon_time = u.run_reconstruction(data, params)
        BDICT[f'tr_25_cuda_tomopy_fbp_stride-{params["undersample"]}'].append(recon_time)

print('\n ##################### \n')
## FBP tomopy ncore=1 ##
params = {
    'undersample': 100,
    'resize_row': RESIZE,
    'thetas': thetas,
    'center': CENTER,
    'algorithm': 'fbp',
    'filter_name': 'ramlak',
    'ncore': 1,
    'circ_mask': CIRC_MASK,
    'save_path': None,  # str(project_root / 'processed_data/0801_tr_lp590_25_recon.npy'),
    'plot': True,
    'plot_title': 'FBP tomopy TR ncore=1 25 Reconstruction'
}

recon_time = u.run_reconstruction(data, params)

if RUN_BENCHMARKS:
    BDICT[f'tr_25_tomopy_fbp_cpu_stride-{params["undersample"]}_core-{params["ncore"]}'] = [recon_time]
    params['save_path'] = None
    for i in range(2):
        recon_time = u.run_reconstruction(data, params)
        BDICT[f'tr_25_tomopy_fbp_cpu_stride-{params["undersample"]}_core-{params["ncore"]}'].append(recon_time)

print('\n ##################### \n')
## FBP tomopy ncore=8 ##
params = {
    'undersample': 1,
    'resize_row': RESIZE,
    'thetas': thetas,
    'center': CENTER,
    'algorithm': 'fbp',
    'filter_name': 'ramlak',
    'ncore': 8,
    'circ_mask': CIRC_MASK,
    'save_path': str(project_root / 'benchmarks/results/0801_tr_lp590_25_cpu_recon.npy'),
    'plot': True,
    'plot_title': 'FBP tomopy TR ncore=8 25 Reconstruction'
}

recon_time = u.run_reconstruction(data, params)

if RUN_BENCHMARKS:
    BDICT[f'tr_25_tomopy_fbp_cpu_stride-{params["undersample"]}_core-{params["ncore"]}'] = [recon_time]
    params['save_path'] = None
    for i in range(2):
        recon_time = u.run_reconstruction(data, params)
        BDICT[f'tr_25_tomopy_fbp_cpu_stride-{params["undersample"]}_core-{params["ncore"]}'].append(recon_time)
# sys.exit(0)  # Exit after processing 25 steps to avoid running the rest of the code


###########################
## 50 step Transmission ##
###########################
print('#########################################################')
print("Processing 50 step Transmission data...")
# 1. Load data from save path in raw2clean.py
data, thetas = u.load_data(project_root / 'processed_data/0801_tr_lp590_50_clean.npy')

print('\n ##################### \n')
## FBP CUDA from astra toolbox ##
#################################
params = {
    'undersample': 1,
    'resize_row': RESIZE,
    'thetas': thetas,
    'center': CENTER,
    'algorithm': tom.astra,
    'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA'},
    'ncore': 1,
    'circ_mask': CIRC_MASK,
    'save_path': str(project_root / 'benchmarks/results/0801_tr_lp590_50_recon.npy'),
    'plot': True,
    'plot_title': 'FBP CUDA TR 50 Reconstruction'
}

recon_time = u.run_reconstruction(data, params)

if RUN_BENCHMARKS:
    BDICT[f'tr_50_cuda_tomopy_fbp_stride-{params["undersample"]}'] = [recon_time]
    params['save_path'] = None
    for i in range(2):
        recon_time = u.run_reconstruction(data, params)
        BDICT[f'tr_50_cuda_tomopy_fbp_stride-{params["undersample"]}'].append(recon_time)


print('\n ##################### \n')
## FBP tomopy ncore=1 ##
###################################
params = {
    'undersample': 100,
    'resize_row': RESIZE,
    'thetas': thetas,
    'center': CENTER,
    'algorithm': 'fbp',
    'filter_name': 'ramlak',
    'ncore': 1,
    'circ_mask': CIRC_MASK,
    'save_path': None,  # str(project_root / 'processed_data/0801_tr_lp590_50_recon.npy'),
    'plot': True,
    'plot_title': 'FBP tomopy TR ncore=1 50 Reconstruction'
}

recon_time = u.run_reconstruction(data, params)

if RUN_BENCHMARKS:
    BDICT[f'tr_50_tomopy_fbp_cpu_stride-{params["undersample"]}_core-{params["ncore"]}'] = [recon_time]
    params['save_path'] = None
    for i in range(2):
        recon_time = u.run_reconstruction(data, params)
        BDICT[f'tr_50_tomopy_fbp_cpu_stride-{params["undersample"]}_core-{params["ncore"]}'].append(recon_time)


print('\n ##################### \n')
## FBP tomopy ncore=8 ##
###################################
params = {
    'undersample': 1,
    'resize_row': RESIZE,
    'thetas': thetas,
    'center': CENTER,
    'algorithm': 'fbp',
    'filter_name': 'ramlak',
    'ncore': 8,
    'circ_mask': CIRC_MASK,
    'save_path': str(project_root / 'benchmarks/results/0801_tr_lp590_50_cpu_recon.npy'),
    'plot': True,
    'plot_title': 'FBP tomopy TR ncore=8 50 Reconstruction'
}

recon_time = u.run_reconstruction(data, params)

if RUN_BENCHMARKS:
    BDICT[f'tr_50_tomopy_fbp_cpu_stride-{params["undersample"]}_core-{params["ncore"]}'] = [recon_time]
    params['save_path'] = None
    for i in range(2):
        recon_time = u.run_reconstruction(data, params)
        BDICT[f'tr_50_tomopy_fbp_cpu_stride-{params["undersample"]}_core-{params["ncore"]}'].append(recon_time)


###########################
## 400 step Transmission ##
###########################
print('#########################################################')
print("Processing 400 step Transmission data...")
# 1. Load data from save path in raw2clean.py
data, thetas = u.load_data(project_root / 'processed_data/0801_tr_lp590_400_clean.npy')

## FBP CUDA from astra toolbox ##
#################################
params = {
    'undersample': 1,
    'resize_row': RESIZE,
    'thetas': thetas,
    'center': CENTER,
    'algorithm': tom.astra,
    'options': {'proj_type': 'cuda', 'method': 'FBP_CUDA'},
    'ncore': 1,
    'circ_mask': CIRC_MASK,
    'save_path': str(project_root / 'benchmarks/results/0801_tr_lp590_400_recon.npy'),
    'plot': True,
    'plot_title': 'FBP CUDA TR 400 Reconstruction'
}

recon_time = u.run_reconstruction(data, params)

if RUN_BENCHMARKS:
    BDICT[f'tr_400_cuda_tomopy_fbp_stride-{params["undersample"]}'] = [recon_time]
    params['save_path'] = None
    for i in range(2):
        recon_time = u.run_reconstruction(data, params)
        BDICT[f'tr_400_cuda_tomopy_fbp_stride-{params["undersample"]}'].append(recon_time)


if RUN_BENCHMARKS:
    # save the BDICT to analysis folder
    with open(str(project_root / 'benchmarks/results/benchmarks_tr.npy'), 'wb') as f:
        np.save(f, BDICT)

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
## 25 step FLuorescence ##
###########################
print('######################################################')
print("###### Processing 25 step FLuorescence data... #######")
# 1. Load data from save path in raw2clean.py
data, thetas = u.load_data(project_root / 'processed_data/0801_fl_lp590_25_clean.npy')

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
    'save_path': str(project_root / 'benchmarks/results/0801_fl_lp590_25_recon.npy'),
    'plot': True,
    'plot_title': 'FBP CUDA FL 25 Reconstruction'
}

recon_time = u.run_reconstruction(data, params)

if RUN_BENCHMARKS:
    BDICT[f'fl_25_cuda_tomopy_fbp_stride-{params["undersample"]}'] = [recon_time]
    params['save_path'] = None
    for i in range(2):
        recon_time = u.run_reconstruction(data, params)
        BDICT[f'fl_25_cuda_tomopy_fbp_stride-{params["undersample"]}'].append(recon_time)


print('\n ##################### \n')
## SART CUDA from astra toolbox ##
params = {
    'undersample': 100,
    'resize_row': RESIZE,
    'thetas': thetas,
    'center': CENTER,
    'algorithm': tom.astra,
    'options': {'method':'SART', 'num_iter':200, 'proj_type':'linear','extra_options':{'MinConstraint':0}},
    'circ_mask': CIRC_MASK,
    'save_path': str(project_root / 'benchmarks/results/0801_fl_lp590_25_sart_recon.npy'),
    'plot': True,
    'plot_title': 'SART CUDA FL 25 Reconstruction'
}

recon_time = u.run_reconstruction(data, params)

if RUN_BENCHMARKS:
    BDICT[f'fl_25_cuda_tomopy_sart_stride-{params["undersample"]}'] = [recon_time]
    params['save_path'] = None
    for i in range(2):
        recon_time = u.run_reconstruction(data, params)
        BDICT[f'fl_25_cuda_tomopy_sart_stride-{params["undersample"]}'].append(recon_time)
# sys.exit(0)


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
    'save_path': None,  # str(project_root / 'benchmarks/results/0801_fl_lp590_25_recon.npy'),
    'plot': True,
    'plot_title': 'FBP tomopy FL ncore=1 25 Reconstruction'
}

recon_time = u.run_reconstruction(data, params)

if RUN_BENCHMARKS:
    BDICT[f'fl_25_tomopy_fbp_cpu_stride-{params["undersample"]}_core-{params["ncore"]}'] = [recon_time]
    params['save_path'] = None
    for i in range(2):
        recon_time = u.run_reconstruction(data, params)
        BDICT[f'fl_25_tomopy_fbp_cpu_stride-{params["undersample"]}_core-{params["ncore"]}'].append(recon_time)

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
    'save_path': str(project_root / 'benchmarks/results/0801_fl_lp590_25_cpu_recon.npy'),
    'plot': True,
    'plot_title': 'FBP tomopy FL ncore=8 25 Reconstruction'
}

recon_time = u.run_reconstruction(data, params)

if RUN_BENCHMARKS:
    BDICT[f'fl_25_tomopy_fbp_cpu_stride-{params["undersample"]}_core-{params["ncore"]}'] = [recon_time]
    params['save_path'] = None
    for i in range(2):
        recon_time = u.run_reconstruction(data, params)
        BDICT[f'fl_25_tomopy_fbp_cpu_stride-{params["undersample"]}_core-{params["ncore"]}'].append(recon_time)
# sys.exit(0)  # Exit after processing 25 steps to avoid running the rest of the code


###########################
## 50 step FLuorescence ##
###########################
print('######################################################')
print("###### Processing 50 step FLuorescence data... ######")
# 1. Load data from save path in raw2clean.py
data, thetas = u.load_data(project_root / 'processed_data/0801_fl_lp590_50_clean.npy')

print('\n ##################### \n')
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
    'save_path': str(project_root / 'benchmarks/results/0801_fl_lp590_50_recon.npy'),
    'plot': True,
    'plot_title': 'FBP CUDA FL 50 Reconstruction'
}

recon_time = u.run_reconstruction(data, params)

if RUN_BENCHMARKS:
    BDICT[f'fl_50_cuda_tomopy_fbp_stride-{params["undersample"]}'] = [recon_time]
    params['save_path'] = None
    for i in range(2):
        recon_time = u.run_reconstruction(data, params)
        BDICT[f'fl_50_cuda_tomopy_fbp_stride-{params["undersample"]}'].append(recon_time)


print('\n ##################### \n')
## SART CUDA from astra toolbox ##
params = {
    'undersample': 100,
    'resize_row': RESIZE,
    'thetas': thetas,
    'center': CENTER,
    'algorithm': tom.astra,
    'options': {'method':'SART', 'num_iter':200, 'proj_type':'linear','extra_options':{'MinConstraint':0}},
    'circ_mask': CIRC_MASK,
    'save_path': str(project_root / 'benchmarks/results/0801_fl_lp590_50_sart_recon.npy'),
    'plot': True,
    'plot_title': 'SART CUDA FL 50 Reconstruction'
}

recon_time = u.run_reconstruction(data, params)

if RUN_BENCHMARKS:
    BDICT[f'fl_50_cuda_tomopy_sart_stride-{params["undersample"]}'] = [recon_time]
    params['save_path'] = None
    for i in range(2):
        recon_time = u.run_reconstruction(data, params)
        BDICT[f'fl_50_cuda_tomopy_sart_stride-{params["undersample"]}'].append(recon_time)


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
    'save_path': None,  # str(project_root / 'benchmarks/results/0801_fl_lp590_50_recon.npy'),
    'plot': True,
    'plot_title': 'FBP tomopy FL ncore=1 50 Reconstruction'
}

recon_time = u.run_reconstruction(data, params)

if RUN_BENCHMARKS:
    BDICT[f'fl_50_tomopy_fbp_cpu_stride-{params["undersample"]}_core-{params["ncore"]}'] = [recon_time]
    params['save_path'] = None
    for i in range(2):
        recon_time = u.run_reconstruction(data, params)
        BDICT[f'fl_50_tomopy_fbp_cpu_stride-{params["undersample"]}_core-{params["ncore"]}'].append(recon_time)


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
    'save_path': str(project_root / 'benchmarks/results/0801_fl_lp590_50_cpu_recon.npy'),
    'plot': True,
    'plot_title': 'FBP tomopy FL ncore=8 50 Reconstruction'
}

recon_time = u.run_reconstruction(data, params)

if RUN_BENCHMARKS:
    BDICT[f'fl_50_tomopy_fbp_cpu_stride-{params["undersample"]}_core-{params["ncore"]}'] = [recon_time]
    params['save_path'] = None
    for i in range(2):
        recon_time = u.run_reconstruction(data, params)
        BDICT[f'fl_50_tomopy_fbp_cpu_stride-{params["undersample"]}_core-{params["ncore"]}'].append(recon_time)
# sys.exit(0)  # Exit after processing 50 steps to avoid running the rest of the code

###########################
## 400 step FLuorescence ##
###########################
print('######################################################')
print("###### Processing 400 step FLuorescence data... ######")
# 1. Load data from save path in raw2clean.py
data, thetas = u.load_data(project_root / 'processed_data/0801_fl_lp590_400_clean.npy')

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
    'save_path': str(project_root / 'benchmarks/results/0801_fl_lp590_400_recon.npy'),
    'plot': True,
    'plot_title': 'FBP CUDA FL 400 Reconstruction'
}

recon_time = u.run_reconstruction(data, params)

if RUN_BENCHMARKS:
    BDICT[f'fl_400_cuda_tomopy_fbp_stride-{params["undersample"]}'] = [recon_time]
    params['save_path'] = None
    for i in range(2):
        recon_time = u.run_reconstruction(data, params)
        BDICT[f'fl_400_cuda_tomopy_fbp_stride-{params["undersample"]}'].append(recon_time)


print('\n ##################### \n')
## SART CUDA from astra toolbox ##
params = {
    'undersample': 100,
    'resize_row': RESIZE,
    'thetas': thetas,
    'center': CENTER,
    'algorithm': tom.astra,
    'options': {'method':'SART', 'num_iter':200, 'proj_type':'linear','extra_options':{'MinConstraint':0}},
    'circ_mask': CIRC_MASK,
    'save_path': str(project_root / 'benchmarks/results/0801_fl_lp590_400_sart_recon.npy'),
    'plot': True,
    'plot_title': 'SART CUDA FL 400 Reconstruction'
}

recon_time = u.run_reconstruction(data, params)

if RUN_BENCHMARKS:
    BDICT[f'fl_400_cuda_tomopy_sart_stride-{params["undersample"]}'] = [recon_time]
    params['save_path'] = None
    for i in range(2):
        recon_time = u.run_reconstruction(data, params)
        BDICT[f'fl_400_cuda_tomopy_sart_stride-{params["undersample"]}'].append(recon_time)
    

# save the BDICT to analysis folder
if RUN_BENCHMARKS:
    with open(str(project_root / 'benchmarks/results/benchmarks_fl.npy'), 'wb') as f:
        np.save(f, BDICT)



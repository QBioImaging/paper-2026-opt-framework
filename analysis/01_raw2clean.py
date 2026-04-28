import sys
import os
from pathlib import Path
import gc
import numpy as np

import cv2
from tqdm import tqdm

from tomopy.recon.rotation import find_center_vo, find_center
import tomopy as tom

project_root = Path.cwd().resolve().parent
if not (project_root / "utils").exists():
    raise Exception(f"You have to keep the original repository structure, current project root: {project_root}")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils import utils_opt as u
from utils.data_loader import OptLoader
from utils.correction_class import Correct


def badcorr3D(data, Corr):
    out = np.empty(data.shape, dtype=data.dtype)
    for i, img in tqdm(enumerate(data)):
        out[i] = Corr.correctBadPxs(img)
    return out


def process_data_FL(data, save=True, filename=None):
    print('#### Processing data... ####')
    ## CORRECTION ##
    Corr = Correct(std_mult=7)
    Corr.set_dark(dark=ddark)
    Corr.set_bad(bad=dhot)

    ## DARK and BRIGHT correction ##
    gc.collect()
    data_corr = Corr.correct_dark_bright(
        data,
        modality='Emission',
        useDark=True,
        useBright=False,
    )

    print(f'After DARK and BRIGHT correction max: {np.amax(data_corr)}, min: {np.amin(data_corr)}.')

    Corr.get_bad_pxs()
    print(len(Corr.hot_pxs), len(Corr.dead_pxs))

    data_corr = badcorr3D(data_corr, Corr)
    print(f'After BAD pixel Correction max: {np.amax(data_corr)}, min: {np.amin(data_corr)}.')
    print(f'dtype before saving: {data_corr.dtype}')

    del data
    gc.collect()

    ## INTENSITY correction ##
    data_corr, _ = Corr.correct_int(
        data_corr,
        mode='integral',
        use_bright=False,
        rect_dim=50,
    )
    print(f'After INTENSITY max: {np.amax(data_corr)}, min: {np.amin(data_corr)}.')

    ## FL BLEACHING correction ##
    meanOverColumns = data_corr.mean(axis=2).T
    decay = meanOverColumns.mean(axis=0) / max(meanOverColumns.mean(axis=0))
    data_corr = np.divide(data_corr.T, decay).T.astype(np.float16)

    print(f'After Bleaching correction max: {np.amax(data_corr)}, min: {np.amin(data_corr)}.')

    del decay
    del meanOverColumns
    gc.collect()

    # Save data
    if save and filename:
        print(f'Saving data... as {data_corr.dtype}')
        # data2save = u.data2saveFormat(data_corr)  # this converts to uint16 and range 0-4095
        with open(os.path.join(project_root, f'processed_data/{filename}.npy'), 'wb') as f:
            np.save(f, data_corr)
    else:
        print('Data not saved, as save=False or filename is None.')
    print('#### Data processing complete! ####')


def process_data_TR(data, save=True, filename=None):
    print('#### Processing data... ####')

    ## CORRECTION ##
    Corr = Correct(std_mult=7)
    Corr.set_dark(dark=ddark)
    Corr.set_bad(bad=dhot)
    Corr.set_bright(bright=dflat)

    ## DARK and BRIGHT correction ##
    gc.collect()
    data_corr = Corr.correct_dark_bright(
        data,
        modality='Transmission',
        useDark=True,
        useBright=True,
    )

    print(f'After DARK and BRIGHT correction max: {np.amax(data_corr)}, min: {np.amin(data_corr)}.')

    Corr.get_bad_pxs()
    print(len(Corr.hot_pxs), len(Corr.dead_pxs))

    data_corr = badcorr3D(data_corr, Corr)
    print(f'After BAD pixel Correction max: {np.amax(data_corr)}, min: {np.amin(data_corr)}.')

    del data
    gc.collect()
    if np.sum(data_corr == 0):
        print('There are zeros in the data, I replace them with 1.')
        data_corr[data_corr == 0] = 1
    
    data_corr = tom.minus_log(data_corr, 2).astype(np.float16)  # two cores

    # Save data
    if save and filename:
        print(f'Saving data...{data_corr.dtype}')
        # data2save = u.data2saveFormat(data_corr)
        with open(os.path.join(project_root, f'processed_data/{filename}.npy'), 'wb') as f:
            np.save(f, data_corr)
    else:
        print('Data not saved, as save=False or filename is None.')
    print('#### Data processing complete! ####')


## Fluorescence 25 steps ##
###########################
# because center pixel is at 779

# Loading data
folder = project_root.joinpath('raw_data', '2024_08_01-fluorescence/2024_08_01-11-39-09_fl_25')
folder_corr = project_root.joinpath('raw_data', '2024_08_01-fluorescence')

ddark = cv2.imread(str(folder_corr.joinpath('../2024_08_01-transmission/2024_08_01-12-02-29_dark_field.tiff')), cv2.IMREAD_UNCHANGED)
dhot = cv2.imread(str(folder_corr.joinpath('2024_08_01-12-03-54_corr_hot.tiff')), cv2.IMREAD_UNCHANGED)

opt = OptLoader(folder, depth=np.int16, format='tiff')  # Load folder
opt.load_folder(mode='simple', stride=1)

process_data_FL(opt.data, save=True, filename='0801_fl_lp590_25_clean')
# sys.exit(0)

## Fluorescence 50 steps ##
###########################
# because center pixel is at 779

# Loading data
folder = project_root.joinpath('raw_data', '2024_08_01-fluorescence/2024_08_01-11-36-16_fl_50')
folder_corr = project_root.joinpath('raw_data', '2024_08_01-fluorescence')

ddark = cv2.imread(str(folder_corr.joinpath('../2024_08_01-transmission/2024_08_01-12-02-29_dark_field.tiff')), cv2.IMREAD_UNCHANGED)
dhot = cv2.imread(str(folder_corr.joinpath('2024_08_01-12-03-54_corr_hot.tiff')), cv2.IMREAD_UNCHANGED)

opt = OptLoader(folder, depth=np.int16, format='tiff')  # Load folder
opt.load_folder(mode='simple', stride=1)

process_data_FL(opt.data, save=True, filename='0801_fl_lp590_50_clean')


## Fluorescence 400 steps ##
############################
# center pixel is at 779

# Loading data
folder = project_root.joinpath('raw_data', '2024_08_01-fluorescence/2024_08_01-11-22-21_fl_400')
folder_corr = project_root.joinpath('raw_data', '2024_08_01-fluorescence')

ddark = cv2.imread(str(folder_corr.joinpath('../2024_08_01-transmission/2024_08_01-12-02-29_dark_field.tiff')), cv2.IMREAD_UNCHANGED)
dhot = cv2.imread(str(folder_corr.joinpath('2024_08_01-12-03-54_corr_hot.tiff')), cv2.IMREAD_UNCHANGED)

opt = OptLoader(folder, depth=np.int16, format='tiff')  # Load folder
opt.load_folder(mode='simple', stride=1)

process_data_FL(opt.data, save=True, filename='0801_fl_lp590_400_clean')
# sys.exit(0)

###########################
###########################
## Transmission 25 steps ##
###########################
# Loading data
folder = project_root.joinpath('raw_data', '2024_08_01-transmission/2024_08_01-11-49-56_tr_25')
folder_corr = project_root.joinpath('raw_data', '2024_08_01-transmission')

dflat = cv2.imread(str(folder_corr.joinpath('2024_08_01-12-01-01_flat_field.tiff')), cv2.IMREAD_UNCHANGED)
ddark = cv2.imread(str(folder_corr.joinpath('2024_08_01-12-02-29_dark_field.tiff')), cv2.IMREAD_UNCHANGED)
dhot = cv2.imread(str(folder_corr.joinpath('2024_08_01-12-03-54_corr_hot.tiff')), cv2.IMREAD_UNCHANGED)

opt = OptLoader(folder, depth=np.int16, format='tiff')  # Load folder
opt.load_folder(mode='simple', stride=1)

process_data_TR(opt.data, save=True, filename='0801_tr_lp590_25_clean')


## Transmission 50 steps ##
###########################
# Loading data
folder = project_root.joinpath('raw_data', '2024_08_01-transmission/2024_08_01-11-48-43_tr_50')
folder_corr = project_root.joinpath('raw_data', '2024_08_01-transmission')

dflat = cv2.imread(str(folder_corr.joinpath('2024_08_01-12-01-01_flat_field.tiff')), cv2.IMREAD_UNCHANGED)
ddark = cv2.imread(str(folder_corr.joinpath('2024_08_01-12-02-29_dark_field.tiff')), cv2.IMREAD_UNCHANGED)
dhot = cv2.imread(str(folder_corr.joinpath('2024_08_01-12-03-54_corr_hot.tiff')), cv2.IMREAD_UNCHANGED)

opt = OptLoader(folder, depth=np.int16, format='tiff')  # Load folder
opt.load_folder(mode='simple', stride=1)

process_data_TR(opt.data, save=True, filename='0801_tr_lp590_50_clean')


## Transmission 400 steps ##
###########################
# Loading data
folder = project_root.joinpath('raw_data', '2024_08_01-transmission/2024_08_01-11-45-31_tr_400')
folder_corr = project_root.joinpath('raw_data', '2024_08_01-transmission')

dflat = cv2.imread(str(folder_corr.joinpath('2024_08_01-12-01-01_flat_field.tiff')), cv2.IMREAD_UNCHANGED)
ddark = cv2.imread(str(folder_corr.joinpath('2024_08_01-12-02-29_dark_field.tiff')), cv2.IMREAD_UNCHANGED)
dhot = cv2.imread(str(folder_corr.joinpath('2024_08_01-12-03-54_corr_hot.tiff')), cv2.IMREAD_UNCHANGED)

opt = OptLoader(folder, depth=np.int16, format='tiff')  # Load folder
opt.load_folder(mode='simple', stride=1)

process_data_TR(opt.data, save=True, filename='0801_tr_lp590_400_clean')


## Transmission 800 steps ##
###########################
# Loading data
folder = project_root.joinpath('raw_data', '2024_08_01-transmission/2024_08_01-11-50-51_tr_800')
folder_corr = project_root.joinpath('raw_data', '2024_08_01-transmission')

dflat = cv2.imread(str(folder_corr.joinpath('2024_08_01-12-01-01_flat_field.tiff')), cv2.IMREAD_UNCHANGED)
ddark = cv2.imread(str(folder_corr.joinpath('2024_08_01-12-02-29_dark_field.tiff')), cv2.IMREAD_UNCHANGED)
dhot = cv2.imread(str(folder_corr.joinpath('2024_08_01-12-03-54_corr_hot.tiff')), cv2.IMREAD_UNCHANGED)

opt = OptLoader(folder, depth=np.int16, format='tiff')  # Load folder
opt.load_folder(mode='simple', stride=1)

process_data_TR(opt.data, save=True, filename='0801_tr_lp590_800_clean')
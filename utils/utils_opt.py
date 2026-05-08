from enum import Enum
import os
import numpy as np
import warnings
from scipy import ndimage as ndi
from tomopari.processors.OPTProcessor import OPTProcessor

import gc
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import tomopy as tom

from tqdm import tqdm
from skimage.transform import resize
from skimage.segmentation import chan_vese

from napari_opt_handler.corrections import Correct
from pympler import muppy, summary


def memory_profile():
    all_objects = muppy.get_objects()
    summary.print_(summary.summarize(all_objects))

#################
# Normalization #
#################
def norm2d(arr: np.ndarray) -> np.ndarray:
    """Normalize a 2D array to the range 0-4095 as uint16."""
    mn = np.amin(arr)
    mx = np.amax(arr)
    return ((arr - mn)/(mx-mn)*4095).astype(np.uint16)


def norm_max(img):
    """ normalize by division by maximum """
    return img/np.amax(img)


###############
# Corrections #
###############
def badcorr3D(data: np.ndarray, corr: Correct) -> np.ndarray:
    out = np.empty(data.shape)
    for i, img in tqdm(enumerate(data)):
        out[i] = corr.correctBadPxs(img)
    return out


######################
# Plotting functions #
######################
def histogram(arr: np.ndarray, name: str, hist_dict:dict = None, bins: int = 256, plot: bool = False) -> dict:
    """Compute and optionally plot the histogram of an array.

    Args:
        arr (np.ndarray): Input array.
        name (str): Name for the histogram.
        hist_dict (dict, optional): Dictionary to store histograms. Defaults to None.
        bins (int, optional): Number of bins for the histogram. Defaults to 256.
        plot (bool, optional): Whether to plot the histogram. Defaults to False.
    
    Returns:
        dict: Updated histogram dictionary.
    """
    hist, bin_edges = np.histogram(arr, bins=bins)
    if plot:
        plt.figure()
        plt.title(name)
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="black")
        plt.yscale('log')
        plt.show()
    if hist_dict is not None:
        hist_dict[name] = (hist, bin_edges)
    else:
        hist_dict = {name: (hist, bin_edges)}
    return hist_dict


def plot_histograms(hist_dict: dict) -> None:
    """Plot multiple histograms from a dictionary.
    
    Args:
        hist_dict (dict): Dictionary containing histograms.
    """
    plt.figure()
    for name, (hist, bin_edges) in hist_dict.items():
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.plot(bin_edges[:-1], hist, label=name)
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()


def move_up(ax, dy):
    pos = ax.get_position()
    ax.set_position([
        pos.x0,
        pos.y0 + dy,
        pos.width,
        pos.height
    ])

def move_horizontal(ax, dx):
    pos = ax.get_position()
    ax.set_position([
        pos.x0 + dx,
        pos.y0,
        pos.width,
        pos.height
    ])


def plot_histograms_paper(
    ax,
    hist_dict: dict,
    *,
    xlabel: str = "Pixel Value",
    ylabel: str = "Frequency",
    logy: bool = True,
    logx: bool = False,
    grid: bool = True,
):
    """Plot multiple histograms on a given Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to draw on.
    hist_dict : dict
        {name: (hist, bin_edges)}
    """
    for name, (hist, bin_edges) in hist_dict.items():
        ax.plot(bin_edges[:-1], hist, label=name)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    if grid:
        ax.grid(True, which="both", ls="--", alpha=0.4)

    ax.legend()


def plot_recon(recon: np.ndarray, plot_path: str = None, title: str = 'Reconstruction slices') -> None:
    """ Plot some slices of the reconstruction
    4x3 grid of images from the reconstruction stack.

    Args:
        recon (np.ndarray): 3D array of reconstruction data.
        plot_path (str, optional): Path to save the plot. Defaults to None.
        title (str, optional): Title of the plot. Defaults to 'Reconstruction slices'.
    """
    height = recon.shape[0]
    _, ax = plt.subplots(4, 3, figsize=(8, 14), sharex=True, sharey=True)
    lineidx = []
    print('min max of reconstructions:',
      np.amin(recon),
      np.amax(recon))

    for i in range(len(recon)):
        try:
            ax[i//3, i%3].imshow(recon[int(height/20*i)], cmap=plt.cm.Greys_r)
            lineidx.append(int(height/20*i))
        except:
            pass
    plt.suptitle(title)
    plt.tight_layout()
    if plot_path is not None:
        print(f"Saving plot to {plot_path}")
        plt.savefig(plot_path)

    plt.show()


###########################
# Reconstruction function #
###########################
class Rec_Modes(Enum):
    """Supported reconstruction modes."""
    FBP_CPU = 0
    FBP_GPU = 1
    TWIST_CPU = 2
    TOMODL_CPU = 3
    TOMODL_GPU = 4
    UNET_CPU = 5
    UNET_GPU = 6


def reconstruct_tomopari(
    sinogram: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, float]:
    """
    Reconstructs a sinogram using the OPTProcessor.
    """
    opt = OPTProcessor()
    opt.rec_process = params.get("method", Rec_Modes.FBP_CPU)
    opt.filter_name = params.get("filter", None)
    opt.use_filter = True if opt.filter_name is not None else False
    opt.batch_size = params.get("batch_process", 4)
    opt.is_half_rotation = params.get("is_half_rotation", False)
    opt.order_mode = params.get("order_mode", 0)
    opt.clip_to_circle = params.get("clip_to_circle", False)
    opt.set_reconstruction_process()
    opt.invert_color = params.get("invert_color", False)

    undersample = params.get('undersample', 1)
    circ_mask = params.get('circ_mask', 0.95)
    save_path = params.get('save_path', None)
    plot = params.get('plot', False)
    plot_title = params.get('plot_title', 'Reconstruction')

    if opt.order_mode == 0:
        sinogram = np.moveaxis(sinogram, 1, 2)
        if undersample > 1:
            sinogram = sinogram[:, :, ::undersample]
        opt.theta, Q, Z = sinogram.shape
        print(f"Sinogram shape after undersampling: {sinogram.shape}")
    else:
        raise NotImplementedError(
            "Only order_mode 0 is implemented, which is the default for OPTProcessor",
        )

    recon = []
    if params.get("resize_row", None) is not None:
        opt.resize_val = params["resize_row"]
        sinogram = opt.resize(sinogram)
        print(f"Data shape after resizing: {sinogram.shape}")
        center_shift = opt.resize_val / 2 - params["center"] / Q * opt.resize_val
    else:
        center_shift = Q / 2 - params["center"] / Q

    if abs(center_shift) > 1e-3:
        sinogram = ndi.shift(sinogram, (0, center_shift, 0), mode="nearest")

    slice_reconstruction = range(Z)
    batch_start = slice_reconstruction[0]
    batch_end = batch_start + opt.batch_size
    begin_time = perf_counter()
    while batch_start <= slice_reconstruction[-1]:
        print("Reconstructing slices {} to {}".format(batch_start, batch_end), end="\r")
        zidx = slice(batch_start, batch_end)
        sino_batch = sinogram[:, :, zidx]
        # print(f"Batch shape before processing: {sino_batch.shape}")
        if opt.order_mode == 0:
            sino_batch = sino_batch.transpose(1, 0, 2)
        reconstruction = opt.reconstruct(sino_batch)
        recon.append(reconstruction)
        batch_start = batch_end
        batch_end += opt.batch_size
    end_time = perf_counter()

    del sinogram
    gc.collect()

    recon = np.concatenate(recon, axis=-1)
    recon = np.rollaxis(recon, -1)
    recon = tom.circ_mask(recon, axis=0, ratio=circ_mask).astype(np.float16)
    print(f"Reconstruction time: {end_time-begin_time} seconds")
    print(f"Reconstruction shape: {recon.shape}, dtype: {recon.dtype}")
    print(f"Reconstruction (min, max): {recon.min()}, {recon.max()}")

    if plot and save_path is not None:
        plot_path = save_path.replace('.npy', '_plot.png')
        plot_recon(recon, plot_path, plot_title)

    if save_path is not None:
        params_path = save_path.replace('.npy', '_params.npy')

        np.save(params_path, params)
        print(f"Parameters saved to {params_path}")

        np.save(save_path, recon)
        print(f"Reconstruction saved to {save_path}")

        del recon
        gc.collect()
        return None, end_time - begin_time

    return recon, end_time - begin_time


def run_reconstruction(data, params):
    """
    params: dict with keys:
        - thetas: ndarray, projection angles
        - center: float, rotation center
        - algorithm: str or callable, tomopy algorithm
        - options: dict, options for tomopy (optional)
        - ncore: int, number of cores (optional)
        - circ_mask: float, mask ratio (optional, default 0.95)
        - save_path: str, where to save npy (optional)
        - plot: bool, whether to plot (optional)
        - plot_title: str, title for plot (optional)

    Returns:
        tuple[np.ndarray | None, float]:
            (recon, elapsed_seconds). If save_path is provided, recon is None
            after saving to disk. Otherwise, recon is returned in memory.
    """
    undersample = params.get('undersample', 1)
    resize_row = params.get('resize_row', None)
    thetas = params['thetas']
    center = params['center']
    algorithm = params.get('algorithm', 'fbp')
    filter_name = params.get('filter_name', 'ramlak')
    options = params.get('options', None)
    ncore = params.get('ncore', 1)
    circ_mask = params.get('circ_mask', 0.95)
    save_path = params.get('save_path', None)
    plot = params.get('plot', False)
    plot_title = params.get('plot_title', 'Reconstruction')

    if resize_row is not None:
        width = data.shape[2]
        data = resize(data, (data.shape[0], data.shape[1], resize_row))
        print(f"Data shape after resizing: {data.shape}")

        # CAREFUL, center pixel needs to be changed too
        # center = int(center / width * resize_row)
        center = center / width * resize_row

    # this is useful for CPU FBB, otherwise it will be too slow
    if undersample > 1:
        data = data[:, ::undersample, :]

    if options is not None:
        begin_time = perf_counter()
        recon = tom.recon(
            data,
            thetas,
            center=center,
            algorithm=algorithm,
            options=options,
            ncore=ncore,
        )
        end_time = perf_counter()
    else:
        begin_time = perf_counter()
        print(f'No astra, {algorithm}, {filter_name}')
        recon = tom.recon(
            data,
            thetas,
            center=center,
            algorithm=algorithm,
            filter_name=filter_name,
            ncore=ncore,
        )
        end_time = perf_counter()

    del data
    gc.collect()

    recon = tom.circ_mask(recon, axis=0, ratio=circ_mask).astype(np.float16)
    print(f"Reconstruction time: {end_time-begin_time} seconds")
    print(f"Reconstruction shape: {recon.shape}, dtype: {recon.dtype}")
    print(f"Reconstruction (min, max): {recon.min()}, {recon.max()}")

    # normalize to uint16, I do not want, clean data are float16
    # recon = (recon - np.amin(recon)) / (np.amax(recon) - np.amin(recon))
    # recon = np.multiply(recon, 65535, out=recon, casting='unsafe')
    # print(f'Reconstruction (min, max): {recon.min()}, {recon.max()}')
    # recon = recon.astype(np.uint16, copy=False)
    # print(f"Reconstruction shape: {recon.shape}, dtype: {recon.dtype}")

    if plot and save_path is not None:
        plot_path = save_path.replace('.npy', '_plot.png')
        plot_recon(recon, plot_path, plot_title)

    if save_path is not None:
        params_path = save_path.replace('.npy', '_params.npy')

        np.save(params_path, params)
        print(f"Parameters saved to {params_path}")

        np.save(save_path, recon)
        print(f"Reconstruction saved to {save_path}")

        del recon
        gc.collect()
        return None, end_time - begin_time

    return recon, end_time - begin_time


###########################
# Saving and loading data #
###########################
def load_data(data_path: str) -> tuple[np.ndarray, np.ndarray]:
    """ Load data and calculate thetas

    Args:
        data_path (str): path to the data

    Returns:
        tuple[np.ndarray, np.ndarray]: data and thetas
    """
    data = np.load(data_path)
    n_steps, _, _ = data.shape
    thetas = calc_thetas(n_steps, half=False)
    return data, thetas


def data2saveFormat(data: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    """ Convert data to save format, either 8 or 16 bit depth

    Args:
        data (np.ndarray): input data
        bit_depth (int, optional): bit depth, either 8 or 16. Defaults to 16.

    Returns:
        np.ndarray: converted data
    """
    mn, mx = np.amin(data), np.amax(data)
    if bit_depth == 16:
        data = ((data - mn)/(mx-mn)*4095).astype(np.int16)
    elif bit_depth == 8:
        data = ((data - mn)/(mx-mn)*255).astype(np.int8)
    else:
        raise ValueError('unknown bit parameter value')
    return data


def rename(folder: str) -> None:
    """ Rename files in the folder to a standard format:
    first 9 chars + chars from 14 to 24 + '_' + chars from 9 to 13 + '.tiff'
    Args:
        folder (str): path to the folder
    """
    for name in os.listdir(folder):
        if name[-4:]=='json' or name[-4:]=='tiff':
            continue
        if len(name) <= 10:
            break
        new_name = name[:9]+name[14:24] + '_' + name[9:13] + '.tiff'
        os.rename(Path(folder).joinpath(name), Path(folder).joinpath(new_name))


#####################
# Metrics functions #
#####################
def sharpness_single(img: np.ndarray) -> float:
    """
    Sharpness of img, first normalize and then evaluate sqrt of
    gradients -> average them

    Args:
        img (np.ndarray): 2d image

    Returns:
        float: sharpness value
    """
    norm_img = norm_max(img)
    gy, gx = np.gradient(norm_img)
    gnorm = np.sqrt(gx**2 + gy**2)
    return np.average(gnorm)


def sharpness_stack(data: np.ndarray) -> list[float]:
    """ Run sharpness for every img in the stack

    Args:
        data (np.ndarray): 3d stack of images

    Return:
        list of sharpness values per image
    """
    sharpness = []
    for img in tqdm(data):
        sharpness.append(sharpness_single(img))
    return sharpness


def calc_thetas(steps: int, half=False) -> np.ndarray:
    """ Calculate thetas for reconstruction
    
    Args:
        steps (int): number of projections
        half (bool): half or full scan
    
    Returns:
        np.ndarray: thetas in radians
    """
    if half:
        return np.linspace(0., 180., steps, endpoint=False) / 180. * (2 * np.pi)
    else:
        return np.linspace(0., 360., steps, endpoint=False) / 360. * (2 * np.pi)


def img_to_int_type(img: np.array, dtype: np.dtype = np.int_) -> np.array:
    """ After corrections, resulting array can be dtype float. Two steps are
    taken here. First convert to a chosed dtype and then clip values as if it
    was unsigned int, which the images are.shape

    Args:
        img (np.array): img to convert
        dtype (np.dtype): either np.int8 or np.int16 currently, Defaults to np.int_

    Returns:
        np.array: array as int
    """
    # ans = img.astype(dtype)
    if dtype == np.int8:
        ans = np.clip(img, 0, 255).astype(dtype)
    elif dtype == np.int16:
        ans = np.clip(img, 0, 2**16 - 1).astype(dtype)  # 4095 would be better for 12bit camera
    else:
        ans = np.clip(img, 0, np.amax(img)).astype(np.int_)

    return ans


def is_positive(img: np.ndarray, corr_type='Unknown') -> bool:
    """
    Check if there are negative pixels in the image, if so, warn and return True

    Args:
        img (np.ndarray): image to check
        corr_type (str): type of correction applied, for warning message

    Returns:
        bool: True if there are negative pixels, False otherwise    
    """
    if np.any(img < 0):
        warnings.warn(
            f'{corr_type} correction: Some pixel < 0, casting them to 0.',
            )
        # return for testing purposes, can be better?
        return True
    return False


# Segmentation functions could go here
def segment_data(arr, mu=0.7) -> np.ndarray:
    """
    Segment the data using chan_vese algorithm, which is a level set method
    for image segmentation. The function applies chan_vese to each image
    in the stack and multiplies the original image by the segmentation mask,
    which can help to remove background and enhance features.

    Args:
        arr (np.ndarray): 3D array of images to segment
        mu (float, optional): parameter for chan_vese, higher values make the
            segmentation more smooth. Defaults to 0.

    Returns:
        np.ndarray: segmented data
    """
    out = np.zeros(arr.shape)
    for i, img in tqdm(enumerate(arr)):
        out[i] = img * chan_vese(img, mu=mu)
    return out

import os
import numpy as np
import warnings
import threading
import gc
from time import perf_counter

import matplotlib.pyplot as plt
import tomopy as tom
from tomopy.recon.rotation import find_center_vo

from pathlib import Path
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from skimage.transform import resize
from skimage.segmentation import chan_vese


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
        plt.title(name)
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        # plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), label=name)
        plt.plot(bin_edges[:-1], hist, label=name)
    plt.yscale('log')
    plt.grid()
    plt.legend()
    plt.show()


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
    """
    undesample = params.get('undersample', 1)
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
        center = int(center / width * resize_row)

    # this is useful for CPU FBB, otherwise it will be too slow
    if undesample > 1:
        data = data[:, ::undesample, :]

    if options is not None:
        beg = perf_counter()
        recon = tom.recon(
            data,
            thetas,
            center=center,
            algorithm=algorithm,
            options=options,
            ncore=ncore,
        )
        end = perf_counter()
    else:
        beg = perf_counter()
        print(f'No astra, {algorithm}, {filter_name}')
        recon = tom.recon(
            data,
            thetas,
            center=center,
            algorithm=algorithm,
            filter_name=filter_name,
            ncore=ncore,
        )
        end = perf_counter()
    
    del data
    gc.collect()

    recon = tom.circ_mask(recon, axis=0, ratio=circ_mask).astype(np.float16)
    print(f"Reconstruction time: {end-beg} seconds")

    # CIRCLE or no circle mask
    print(f'Reconstruction (min, max): {recon.min()}, {recon.max()} with type {recon.dtype}')

    # normalize to uint16, I do not want, clean data are float16
    # recon = (recon - np.amin(recon)) / (np.amax(recon) - np.amin(recon))
    # recon = np.multiply(recon, 65535, out=recon, casting='unsafe')
    # print(f'Reconstruction (min, max): {recon.min()}, {recon.max()}')
    # recon = recon.astype(np.uint16, copy=False)
    # print(f"Reconstruction shape: {recon.shape}, dtype: {recon.dtype}")

    if save_path is not None:
        # save bothe the recon and the parameters used
        params_path = save_path.replace('.npy', '_params.npy')
        
        np.save(params_path, params)
        print(f"Parameters saved to {params_path}")
        
        np.save(save_path, recon)
        print(f"Reconstruction saved to {save_path}")

    if plot and save_path is not None:
        plot_path = save_path.replace('.npy', '_plot.png')
        plot_recon(recon, plot_path, plot_title)

    del recon
    gc.collect()
    return end-beg


def run_fbp_thread(
        data: np.ndarray,
        thetas:np.ndarray,
        centers: list[float],
        recon_algo: str = 'art',
    ) -> np.ndarray:
    height = data.shape[1]
    r1 = tom.recon(data[:, height//2:height//2+1, :], thetas,
                   center=centers[height//2],
                   sinogram_order=False,
                   algorithm=recon_algo)
    threads = [None] * height
    data_recon = np.zeros((data.shape[1], *r1.squeeze().shape),
                          dtype=np.float32,
                          )
    start = perf_counter()
    for i in tqdm(range(height)):
        threads[i] = threading.Thread(target=recon_thread,
                                      args=[i, thetas, centers[i],
                                            data, data_recon,
                                            recon_algo],
                                      )
        threads[i].start()

    for i in tqdm(range(len(threads))):
        threads[i].join()
    end = perf_counter()
    print(f'Wall time: {end - start}')
    return data_recon


def recon_thread(idx:int,
                 thetas: np.ndarray,
                 center: float,
                 arr: np.ndarray,
                 arr_out: np.ndarray,
                 recon_algo: str,
                 ) -> None:
    """ Thread function for reconstruction

    Args:
        idx (int): index of the slice to reconstruct
        thetas (np.ndarray): projection angles
        center (float): center of rotation
        arr (np.ndarray): input sinogram array
        arr_out (np.ndarray): output reconstructed array
        recon_algo (str): reconstruction algorithm
    """
    arr_out[idx] = tom.recon(arr[:, idx:idx+1, :],
                             thetas,
                             center=center,
                             sinogram_order=False,
                             algorithm=recon_algo)#.astype(np.int16)


def fbp(original_stack,
        COR: str = 'calc',
        cor_step: int = 100,
        half_angle: bool = False,
        recon_every: int = 1,
        recon_algo: str = 'art',
        ) -> np.ndarray:
    """
    Docstring for fbp function. This function performs filtered back projection (FBP)
    reconstruction on a given stack of images.

    Args:
        original_stack (np.ndarray): The input stack of images for reconstruction.
        COR (str, optional): Center of rotation handling method. Defaults to 'calc'.
        cor_step (int, optional): Step size for center of rotation calculation. Defaults to 100.
        half_angle (bool, optional): Whether to use half-angle reconstruction. Defaults to False.
        recon_every (int, optional): Interval for reconstruction. Defaults to 1.
        recon_algo (str, optional): Reconstruction algorithm to use. Defaults to 'art'.
    
    Returns:
        np.ndarray: The reconstructed image stack.

    """
    print(f'Stack shape, {original_stack.shape}')
    print(f'thread call, {original_stack[:, ::recon_every, :].shape}')
    n_steps, height, _ = original_stack.shape
    thetas = calc_thetas(n_steps, half=half_angle)

    if COR == 'calc':
        print(f'Find COR every {cor_step} pixels vertically')
        center = []
        X = []
        print('Center of rotation')
        for i in tqdm(range(0, int(height / cor_step)-1)):
            X.append(i * cor_step)
            if half_angle:
                cor = find_center_vo(
                                original_stack[:n_steps, :, :],
                                smin=-50, smax=50,
                                ncore=1,
                                ind=i * cor_step,
                                ratio=0.5)
            else:
                cor = find_center_vo(
                                original_stack[:n_steps//2, :, :],
                                smin=-50, smax=50,
                                ncore=1,
                                ind=i * cor_step,
                                ratio=0.5)
            center.append(cor)
        print(np.mean(center), center)
        # linear regression
        lm = LinearRegression()
        lm.fit(np.array(X).reshape(-1, 1), center)
        print(f'coeficients: a={lm.coef_[0]}, b={lm.intercept_}.')
        centers = range(height) * lm.coef_[0] + lm.intercept_
        # print('Running FBP with a mean of the CORs')
        # recon = run_fbp(original_stack, thetas, np.mean(center))
        print('Running FBP with  linearly fitted CORs')
        recon = run_fbp_thread(original_stack[:, ::recon_every, :],
                               thetas,
                               centers[::recon_every],
                               recon_algo=recon_algo)

    elif type(COR) == float:
        print(f'Running FBP with COR const {COR}')
        l = original_stack[:, ::recon_every, :].shape[1]
        recon = run_fbp_thread(original_stack[:, ::recon_every, :],
                               thetas,
                               [COR]*l,
                               recon_algo=recon_algo)
    return recon


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
        ans = ((data - mn)/(mx-mn)*4095).astype(np.int16)
    elif bit_depth == 8:
        ans = ((data - mn)/(mx-mn)*255).astype(np.int8)
    else:
        raise ValueError('unknown bit parameter value')
    return ans


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
    if np.any(img < 0):
        warnings.warn(
            f'{corr_type} correction: Some pixel < 0, casting them to 0.',
            )
        # return for testing purposes, can be better?
        return True
    return False


# Segmentation functions could go here
def segment_data(arr, mu=0.7):
    out = np.zeros(arr.shape)
    for i, img in tqdm(enumerate(arr)):
        out[i] = img * chan_vese(img, mu=mu)
    return out

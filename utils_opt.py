import os
import numpy as np
from tqdm import tqdm
from tomopy.recon.rotation import find_center_vo
from time import perf_counter
from sklearn.linear_model import LinearRegression
from skimage.transform import resize
import threading
import tomopy as tom
from pathlib import Path
import warnings
import gc
import matplotlib.pyplot as plt


def plot_recon(recon, plot_path, title):
    height = recon.shape[0]
    fig, ax = plt.subplots(4, 3, figsize=(8, 14), sharex=True, sharey=True)
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
    else:
        print("Plot path is None, not saving plot.")
    plt.savefig(plot_path)
    plt.show()


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


def load_data(data_path):
    data = np.load(data_path)
    n_steps, _, _ = data.shape
    thetas = calc_thetas(n_steps, half=False)
    return data, thetas


def data2saveFormat(data, bit=16):
    mn, mx = np.amin(data), np.amax(data)
    if bit == 16:
        ans = ((data - mn)/(mx-mn)*4095).astype(np.int16)
    elif bit == 8:
        ans = ((data - mn)/(mx-mn)*255).astype(np.int8)
    else:
        raise ValueError('unknown bit parameter value')
    return ans


def saveVolume():
    pass


def norm_max(img):
    """ normalize by division by maximum """
    return img/np.amax(img)


def sharpness_single(img):
    """ Sharpness of img, first normalize and then evaluate sqrt of
    gradients -> average them
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


def run_fbp_thread(data, thetas, centers, recon_algo):
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


def recon_thread(idx, thetas, center, arr, arr_out, recon_algo):
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
        ):
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


def calc_thetas(steps, half=False):
    if half:
        return np.linspace(0., 180., steps, endpoint=False) / 180. * (2 * np.pi)
    else:
        return np.linspace(0., 360., steps, endpoint=False) / 360. * (2 * np.pi)


def rename(folder):
    for name in os.listdir(folder):
        if name[-4:]=='json' or name[-4:]=='tiff':
            continue
        if len(name) <= 10:
            break
        new_name = name[:9]+name[14:24] + '_' + name[9:13] + '.tiff'
        os.rename(Path(folder).joinpath(name), Path(folder).joinpath(new_name))


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


def is_positive(img, corr_type='Unknown'):
    if np.any(img < 0):
        warnings.warn(
            f'{corr_type} correction: Some pixel < 0, casting them to 0.',
            )
        # return for testing purposes, can be better?
        return 1
    return 0

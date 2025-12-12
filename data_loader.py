import os
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


class OptLoader():
    """
    Takes care about loading opt data from optac experiment
    """
    def __init__(self, folder, depth, format) -> None:
        self.formats = ['tiff']
        self.depths = [np.int8, np.int16, np.uint16]

        self.folder = self.set_folder(folder)
        self.depth = self.set_depth(depth)
        self.file_format = self.set_file_format(format)

    def set_file_format(self, format: str) -> str:
        """
        sets file format of the data images. It should be
        part of experimental metadata. Currently supports jpeg and tiff

        Args:
            format (str): one of the supported formats

        Raises:
            ValueError: If format none of the above

        Returns:
            str: validated/supported file type
        """
        if format not in self.formats:
            raise ValueError(
                'Unsupported file format. See .formats for options',
                )
        return format

    def set_depth(self, depth: np.dtype) -> np.dtype:
        """Set expected depth of the images

        Args:
            depth (np.dtype): Expects int type, currently np.int8 or
             np.int16

        Raises:
            ValueError: If depth not in supported options

        Returns:
            np.dtype: validated depth type
        """
        if depth not in self.depths:
            raise ValueError('Unsupported depth, etiher np.int8 or np.int16')
        return depth

    def set_folder(self, path: str) -> Path:
        """Set folder path, where the data is.

        Args:
            path (str): OS dependent path

        Raises:
            FileNotFoundError: raised if path does not exist

        Returns:
            pathlib.Path: path which exists
        """
        if os.path.exists(Path(path)):
            return Path(path)
        else:
            raise FileNotFoundError('Non-existent path.')

    def load_folder(self, mode='simple', stride=1):
        """ Master method to navigate loading folder methods.
        TODO: The file structure can be detected automatically.

        Args:
            mode (str, optional): Loading method selector.
            Defaults to 'simple'.

        Raises:
            NotImplementedError: Nested loading not implemented
            NotImplementedError: Nested loading + averaging
            ValueError: No valid mode given
        """
        if mode == 'simple':
            self.load_folder_simple(stride)
        # nested dataset, step-wise acquisition, with averages and sweeps
        # can be averaged separately later
        elif mode == 'simplified':
            self.load_folder_simplified(stride)
        elif mode == 'nested':
            raise NotImplementedError
            self.load_folder_nested(average=False)
        elif mode == 'nested_av':
            raise NotImplementedError
            self.load_folder_nested(average=True)
        else:
            raise ValueError('Wrong mode option')

    def load_folder_simple(self, stride):
        """Loading files as a simple sequence. No averaging,
        no nesting of no sweep distinctions. Dile names have to
        follow four digit notation, i.e. 0001.tiff.
        """
        files = [file for file in self.folder.iterdir()
                 if file.name.endswith(self.file_format)]

        # load first file for preallocation
        f0 = np.array(Image.open(files[0]), dtype=self.depth)
        # preallocate
        data = np.empty((len(files)//stride, *f0.shape), dtype=self.depth)
        counter = 0
        for i in tqdm(range(0, len(files), stride)):
            # construct fname
            
            fname = f'{i:04d}.{self.file_format}'
            matches = [str(k) for k in files if fname in str(k)]
            if len(matches) > 1:
                raise ValueError('Got two candidate files, which cannot be.')
            elif matches == []:
                print(files[:4], fname)
                raise ValueError('No matches')
            # load data
            data[counter] = np.array(Image.open(self.folder.joinpath(matches[0])),
                                     dtype=self.depth)
            counter += 1
            # print(i, end='\r')
        self.data = data
        self.n_steps = len(files)

    # this is for sweep type of file format, but lod directly
    def load_folder_simplified(self, stride):
        """Loading files as a simple sequence. No averaging,
        no nesting of no sweep distinctions. Dile names have to
        follow four digit notation, i.e. 0001.tiff.
        """
        files = [file for file in self.folder.iterdir()
                 if file.name.endswith(self.file_format)]

        # load first file for preallocation
        f0 = np.array(Image.open(files[0]), dtype=self.depth)
        # preallocate
        data = np.empty((len(files)//stride, *f0.shape), dtype=self.depth)
        counter = 0
        print(data.shape[0], len(range(0, len(files), stride)))
        # THIS is a bug in the range
        for i in tqdm(range(0, len(files), stride)):
            # construct fname
            fname = f'0_{i}_0.{self.file_format}'
            # load data
            data[counter] = np.array(Image.open(self.folder.joinpath(fname)),
                                     dtype=self.depth)
            counter += 1
        self.data = data
        self.n_steps = len(files)
# paper-2026-opt-framework
Data and codes to reproduce publication results and figures

This README serves two purposes

1. [Reproduce publication figures and data processing](#paper-reproducibility)
2. [Setup guide to the OPT framework](#opt-framework)

## Paper reproducibility
Title: An unified open framework for Optical Projection Tomography: from acquisition to reconstruction
DOI: XXX

### Installation

This setup is using `conda`, 

```bash
# get the repository
git clone https://github.com/QBioImaging/paper-2026-opt-framework.git

conda create -n paper2026-310 python==3.10
conda activate paper2026-310

# enter the repo
cd paper-2026-opt-framework

# install dependencies
pip install -r requirements.txt

# setup a kernel for the NBs
python -m ipykernel install --user --name "opt-framework"
```

further install tomopy

```bash
conda install --channel conda-forge tomopy

# or if you have cuda available
conda install --channel conda-forge tomopy "libtomo=*=cuda*"
```

### Dependencies

The requirements contain standard python image processing libraries and packages developed by us. However for the purpose here, we import only specific modules from those, instead of using then as napari plugins, which aslo demonstrates the flexibility of the source code.

### Data
raw data are deposited at Zenodo (should be bioimage? perhaps), here is [programatic way](https://www.diehlpk.de/blog/zenodo-wget/) to pull the data.

Download the raw_data.tar.gz and open the archive in the `raw_data` folder in this repository. That is why the `raw_data` folder is left empty.

Open archive is 10.4 GB 
```bash
tar -xzvf raw_data.tar.gz -C raw_data/
```

### Notebooks

First notebook `01_process_raw_data`:
- shows a step by step processing pipeline, while saving the processed data to initially empty `processed_data` folder.
- produces Figure XXX of the paper demonstrating 

Second notebook loads raw/processed data to reproduce figure of the paper.

## OPT framework

Links to each tool for the setup is referenced in the article, with detailed instructions in each of the manuals. Here we provide the an ultrafast minimal setup.

### Install ImSwitch OPT

```bash
conda create -n imswitchOTP ''

# install ImSwitch OPT
git clone ...

cd ...

pip install .

cd ..

```

### Install napari and plugins

run napari in the `imswitchOPT` environment

```
python -m napari
```


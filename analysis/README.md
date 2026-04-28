# Analysis

NBs serve for more visual and testing purposes. Beware that the interactivite, ie rerunning of the cells is somewhat limited because the loaded volumes (memory allocation) in case of the fully sampled data are quite large, therefore variables are getting deleted along the workflow, which might result in errors when trying to rerun above your current processing step. The NBs are:

- `01_process_raw_data_fl.ipynb` shows all the loading, correction and reconstruction functionalities for emission tomography data.
- `02_process_raw_data_tr.ipynb` does the similar processsing specific to transmission tomography data.
- `03_fl-tr.ipynb` has a prerequisite of having the reconstructed FL and TR volumes in the `data_output` folder, either you download them or generate them using for example the NBs `01..` and `02...`.

Intermediate outputs, which are the corrected projection stacks are saved into `processed_data` folder, the same as the scripts below. Reconstruction is implemented using FBP from `tomopy` package, which affers CUDA accelaration.

Scripts streamline the processing to exactly reproduce the manuscript analysis. Run `01_raw2clean.py` to generate/rewirte the corercted raw data which are exported to the `processed_data` folder.
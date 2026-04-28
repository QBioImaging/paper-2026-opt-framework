# Data output folder

Data analysis NBs save reconstruction outputs as `.npy` compressed numpy arrays in 16 bit unsigned integers. Full 3D reconstructions are 9.5 GB volumes for each modality:

- 0801_fl_lp590_400_recon.npy
- 0801_tr_lp590_400_recon.npy

These are the final reconstruction volumes. For the reporoducibility reasons, we share also intermediate stacks of corrected pre-processed raw data in the `processed_data` folder. The undersampled reconstructions can be generated from the NBs in the `analyis`, while they are also calculated using different reconstruction methods for the benchmark scripts located in the `benchmarks` folder.
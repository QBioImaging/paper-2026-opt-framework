import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

# load the dictionary
# Fluorescence
project_root = Path.cwd().resolve().parent
with open(project_root / 'benchmarks/results/benchmarks_fl.npy', 'rb') as f:
    BDICT = np.load(f, allow_pickle=True).item()

# calculate average reconstruction times
avg_recon_times = {k: np.mean(v) for k, v in BDICT.items()}
std_recon_times = {k: np.std(v) for k, v in BDICT.items()}
print("Average reconstruction times:")
for k, v in avg_recon_times.items():
    print(f"  {k}: {v:.2f} +- {std_recon_times[k]:.2f} seconds")

# plot the average reconstruction times
plt.barh(list(avg_recon_times.keys()), list(avg_recon_times.values()))
plt.xlabel("Time (seconds)")
plt.title("Average Reconstruction Times, Fluorescence")
plt.tight_layout()
plt.savefig(project_root / 'benchmarks/results/average_reconstruction_times_fl.png')
plt.show()


# load the dictionary
# Transmission
with open(project_root / 'benchmarks/results/benchmarks_tr.npy', 'rb') as f:
    BDICT = np.load(f, allow_pickle=True).item()

# calculate average reconstruction times
avg_recon_times = {k: np.mean(v) for k, v in BDICT.items()}
std_recon_times = {k: np.std(v) for k, v in BDICT.items()}
print("Average reconstruction times:")
for k, v in avg_recon_times.items():
    print(f"  {k}: {v:.2f} +- {std_recon_times[k]:.2f} seconds")

# plot the average reconstruction times
plt.barh(list(avg_recon_times.keys()), list(avg_recon_times.values()))
plt.xlabel("Time (seconds)")
plt.title("Average Reconstruction Times, Transmission")
plt.tight_layout()
plt.savefig(project_root / 'benchmarks/results/average_reconstruction_times_tr.png')
plt.show()

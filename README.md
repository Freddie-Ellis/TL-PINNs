# TL-PINNs
Third year individual project on Transfer Learning (TL) integrated PINNs for fluid mechanics.
I'm currently connecting this repository to MobaXterm for HPC usage.

Workflow is as follows:

Make change in VS -> Push to GitHub -> pull on Moba using 'git pull origin main' from within cd

Make change in Moba -> Push to GitHub ('make the change' -> git status -> git add . -> git commit -m "update message" -> 
git push origin main) -> Pull on local machine in GitHub Desktop

Example test slurm file:

#!/bin/bash

#SBATCH --job-name=test   # Job name
#SBATCH --output=/mainfs/lyceum/fe1g22/TL-PINNs/out/test.out        # Standard output and error log
#SBATCH --time=36:00:00                      # Time limit hrs:min:sec
#SBATCH --partition=lyceum                   # Partition name (adjust as needed)
#SBATCH --nodes=1                            # Number of nodes
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --cpus-per-gpu=8                     # CPU cores per GPU
#SBATCH --mem=32G                            # Memory per node
#SBATCH --mail-type=ALL                      # Mail notifications (BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=fe1g22@soton.ac.uk       # Where to send mail notifications

# Unload all modules to avoid conflicts
module purge

# Load Anaconda 
module load anaconda

# Initialize Conda environment for TensorFlow 2.5.9
source ~/.bashrc
conda activate TENSOR_env  # Activate your environment with TensorFlow 2.5.9

# Optional: Debugging check to confirm GPU visibility and driver compatibility
nvidia-smi

# Run the Python script
python3 -u /mainfs/lyceum/fe1g22/TL-PINNs/PINN_train.py
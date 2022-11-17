#!/bin/bash
#SBATCH --nodes 1             
#SBATCH --gres=gpu:v100l:1         # Request 2 GPU "generic resources”.
#SBATCH --tasks-per-node=1  # Request 1 process per GPU. You will get 1 CPU per process by default. Request more CPUs with the "cpus-per-task" parameter to enable multiple data-loader workers to load data in
#SBATCH --mem=32G      
#SBATCH --time=1-00:00
#SBATCH --output=%N-%j.out

module load python # Using Default Python version - Make sure to choose a version that suits your application
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torchvision --no-index
pip install --no-index scikit-learn
pip install python-hostlist
pip install opencv-python
pip install munkres
pip install PyYAML
pip install numpy --no-index

export NCCL_BLOCKING_WAIT=1  #Set this environment variable if you wish to use the NCCL backend for inter-GPU communication.
export MASTER_ADDR=$(hostname) #Store the master node’s IP address in the MASTER_ADDR environment variable.

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

# The SLURM_NTASKS variable tells the script how many processes are available for this execution. “srun” executes the script <tasks-per-node * nodes> times

srun python main.py
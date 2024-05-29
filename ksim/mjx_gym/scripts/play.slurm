#!/bin/bash
#SBATCH --job-name=robotic-locomotion-inference    # Job name
#SBATCH --output=logs/play_output_%j.log      # Output file
#SBATCH --error=logs/play_error_%j.log        # Error file
#SBATCH --time=72:00:00                           # Maximum execution time
#SBATCH --partition=compute                       # Partition (queue) name
#SBATCH --nodes=1                                 # Number of nodes
#SBATCH --gres=gpu:8                              # Request 8 GPUs per node
#SBATCH --ntasks=2                                # Number of tasks (keep as 2 even if 1 node)
#SBATCH --cpus-per-task=16                        # Number of CPU cores per task

# Activate your Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ksim

# Set environment variables
export MODEL_DIR="../assets"

# Path to the configuration file
CONFIG_PATH="experiments/stompy_walk.yaml"

# Verify allocated GPUs
echo "Nodes allocated:"
scontrol show hostname $SLURM_JOB_NODELIST

# Log GPU utilization every 10 seconds on all nodes
echo "Starting GPU utilization logging"
srun --ntasks=1 --nodes=1 bash -c '
while true; do 
    nvidia-smi --query-gpu=timestamp,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv > logs/gpu_utilization_node_$SLURM_NODEID.log; 
    sleep 10; 
done' &

# Run the inference script
python -u play.py --config $CONFIG_PATH

# Delete temp logs
rm logs/gpu_utilization_temp_*.log

# Deactivate Conda environment
conda deactivate
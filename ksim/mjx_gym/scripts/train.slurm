#!/bin/bash
#SBATCH --job-name=robotic-locomotion-training    # Job name
#SBATCH --output=logs/training_output_%j.log      # Output file
#SBATCH --error=logs/training_error_%j.log        # Error file
#SBATCH --time=24:00:00                           # Maximum execution time
#SBATCH --partition=compute                       # Partition (queue) name
#SBATCH --nodes=2                                 # Number of nodes
#SBATCH --gres=gpu:8                              # Request one GPU

# Activate your Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ksim

# Set path to assets folder
export MODEL_DIR="../assets"

# Path to the configuration file
CONFIG_PATH="experiments/default_humanoid_walk.yaml"

# Log GPU utilization every 10 seconds
while true; do
    nvidia-smi --query-gpu=timestamp,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv >> logs/gpu_utilization.log
    sleep 10
done &

# Run the training script
python -u train.py --config $CONFIG_PATH

# Deactivate Conda environment
conda deactivate

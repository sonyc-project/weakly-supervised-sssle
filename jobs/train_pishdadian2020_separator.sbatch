#!/bin/bash

#BATCH --job-name=train_pishdadian2020_separator
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jtc440@nyu.edu
#SBATCH --output=train_pishdadian2020_separator.out

module purge
source ~/.bashrc

module load sox/intel/14.4.2
module load ffmpeg/intel/3.2.2
module load rubberband/intel/1.8.1
module load cuda/10.1.105
module load cudnn/10.1v7.6.5.32
source activate cdsd

python -u /home/jtc440/dev/class_dependent_spl_decomposition/cdsd/train_separator.py \
    /beegfs/jtc440/cdsd-data/pishdiadian2020-lambda10_soundscapes \
    /home/jtc440/dev/class_dependent_spl_decomposition/resources/training_configs/pishdadian2020_rnn_separator.json \
    /scratch/jtc440/cdsd/models/pishdadian2020_separator \
    --num-data-workers 2 \
    --checkpoint-interval 5

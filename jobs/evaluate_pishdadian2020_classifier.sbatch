#!/bin/bash

#BATCH --job-name=evaluate_pishdadian2020_classifier
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jtc440@nyu.edu
#SBATCH --output=evaluate_pishdadian2020_classifier.out

module purge
source ~/.bashrc

module load sox/intel/14.4.2
module load ffmpeg/intel/3.2.2
module load rubberband/intel/1.8.1
module load cuda/10.1.105
module load cudnn/10.1v7.6.5.32
source activate cdsd

python -u /home/jtc440/dev/class_dependent_spl_decomposition/cdsd/evaluate_classifier.py \
    /beegfs/jtc440/cdsd-data/pishdiadian2020-lambda10_soundscapes \
    /scratch/jtc440/cdsd/models/pishdadian2020_classifier/config.json \
    --num-data-workers 2

#!/bin/bash

#SBATCH --job-name=fsslr4
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:p40:1
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jtc440@nyu.edu
#SBATCH --output=train_pishdadian2020-background_crnn_classifier_frame_unet_separator_frozencls_energyloss_lr1en4_alpha1000.out

set -e

# If there's already a job running, exit
#if [ "$(squeue -u jtc440 | tail -n +2 | grep 'fsslr4' | grep ' R ' | wc -l)" -gt 1 ]; then 
#    exit 0;  
#fi

module purge
source ~/.bashrc

module load sox/intel/14.4.2
module load ffmpeg/intel/3.2.2
module load rubberband/intel/1.8.1
module load cuda/10.1.105
module load cudnn/10.1v7.6.5.32
source activate cdsd

/home/jtc440/miniconda3/envs/cdsd/bin/python -u /home/jtc440/dev/class_dependent_spl_decomposition/cdsd/train_classifier.py \
    /beegfs/jtc440/cdsd-data/pishdadian2020merl-lambda5_soundscapes \
    /home/jtc440/dev/class_dependent_spl_decomposition/resources/training_configs/pishdadian2020-nobackground_crnn_classifier_frame_lr1en4.json \
    /scratch/jtc440/cdsd/models/pishdadian2020-nobackground_crnn_classifier_frame_energyloss_lr1en4_alpha1000 \
    --num-data-workers 4 \
    --checkpoint-interval 5

/home/jtc440/miniconda3/envs/cdsd/bin/python -u /home/jtc440/dev/class_dependent_spl_decomposition/cdsd/train_separator.py \
    /beegfs/jtc440/cdsd-data/pishdadian2020merl-lambda5_soundscapes \
    /home/jtc440/dev/class_dependent_spl_decomposition/resources/training_configs/pishdadian2020-nobackground_unet_separator_crnn_classifier_frame_frozen_energyloss_lr1en4_alpha1000.json \
    /scratch/jtc440/cdsd/models/pishdadian2020-nobackground_unet_separator_crnn_classifier_frame_frozen_energyloss_lr1en4_alpha1000 \
    --num-data-workers 4 \
    --checkpoint-interval 5

/home/jtc440/miniconda3/envs/cdsd/bin/python -u /home/jtc440/dev/class_dependent_spl_decomposition/cdsd/evaluate_separator.py \
    /beegfs/jtc440/cdsd-data/pishdadian2020merl-lambda5_soundscapes \
    /scratch/jtc440/cdsd/models/pishdadian2020-nobackground_unet_separator_crnn_classifier_frame_frozen_energyloss_lr1en4_alpha1000/config.json \
    --num-data-workers 4 \
    --checkpoint earlystopping


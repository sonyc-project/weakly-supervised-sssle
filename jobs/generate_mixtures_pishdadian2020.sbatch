#!/bin/bash

#BATCH --job-name=generate_mixtures_pishdadian2020_classifier
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=20GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jtc440@nyu.edu
#SBATCH --output=generate_mixtures_pishdadian2020.out

set -e

module purge
source ~/.bashrc

module load sox/intel/14.4.2
module load ffmpeg/intel/3.2.2
module load rubberband/intel/1.8.1
source activate cdsd

python /home/jtc440/dev/class_dependent_spl_decomposition/cdsd/generate_mixtures.py \
    /beegfs/jtc440/cdsd-data/us8k-foreground/train \
    /beegfs/jtc440/sonyc_ust_spl_decomposition/background_clips_70/2017 \
    /home/jtc440/dev/class_dependent_spl_decomposition/resources/scaper_specs/pishdadian2020_lambda5.json \
    20000 \
    train \
    /beegfs/jtc440/cdsd-data/pishdiadian2020-lambda5_soundscapes_fixed/train

python /home/jtc440/dev/class_dependent_spl_decomposition/cdsd/generate_mixtures.py \
    /beegfs/jtc440/cdsd-data/us8k-foreground/valid \
    /beegfs/jtc440/sonyc_ust_spl_decomposition/background_clips_70/2017 \
    /home/jtc440/dev/class_dependent_spl_decomposition/resources/scaper_specs/pishdadian2020_lambda5.json \
    5000 \
    valid \
    /beegfs/jtc440/cdsd-data/pishdiadian2020-lambda5_soundscapes_fixed/valid

python /home/jtc440/dev/class_dependent_spl_decomposition/cdsd/generate_mixtures.py \
    /beegfs/jtc440/cdsd-data/us8k-foreground/test \
    /beegfs/jtc440/sonyc_ust_spl_decomposition/background_clips_70/2017 \
    /home/jtc440/dev/class_dependent_spl_decomposition/resources/scaper_specs/pishdadian2020_lambda5.json \
    5000 \
    test \
    /beegfs/jtc440/cdsd-data/pishdiadian2020-lambda5_soundscapes_fixed/test

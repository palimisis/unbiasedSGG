#!/bin/bash 
#SBATCH --partition=luke
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

set -e
# Activate Anaconda work environment 
# source /home/$USER/minicoda3/etc/profile.d/conda.sh
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate TEMPURA


# python3 /home/it21902/unbiasedSGG/train.py $@

# python3 test_cuda.py
python3 test.py -mode predcls \
               -datasize large \
               -data_path /home/it21902/datasets/charades/dataset/ag/ \
               -model_path /home/it21902/unbiasedSGG/output/predcls/best_Mrecall_model.tar \
               -rel_mem_compute joint \
               -rel_mem_weight_type simple \
               -mem_fusion late \
               -mem_feat_selection manual \
               -mem_feat_lambda 0.5  \
               -rel_head gmm \
               -obj_head linear \
               -K 6

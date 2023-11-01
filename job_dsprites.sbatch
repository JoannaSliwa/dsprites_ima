#!/bin/bash -l
#SBATCH -o /cobra/u/jsliwa/Documents/NuX/log/job_%j.out
#SBATCH -e /cobra/u/jsliwa/Documents/NuX/log/job_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jsliwa@mpcdf.mpg.de

conda activate ipcf-env

module load cuda/11.4 cudnn/8.2.4 anaconda/3/2021.11

export XLA_FLAGS=--xla_gpu_cuda_data_dir=/mpcdf/soft/SLE_12/packages/x86_64/cuda/11.4.2.

srun /cobra/u/jsliwa/Documents/NuX/job_dsprites.sh

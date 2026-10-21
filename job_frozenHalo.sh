#!/bin/bash
#SBATCH --job-name=frozenHalo
#SBATCH --account=gpu
#SBATCH --output=/home/jdarne1/Eberhardt_Massive_Stars/SchroPo_public/log/%j.out
#SBATCH --error=/home/jdarne1/Eberhardt_Massive_Stars/SchroPo_public/log/%j.err
#SBATCH --qos=long
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-gpu=6
#SBATCH --mail-user=jdarne@outlook.com
#SBATCH --mail-type=END,FAIL

export CUDA_DEVICE_ORDER=PCI_BUS_ID


# activate python environment
source /home/jdarne1/anaconda3/etc/profile.d/conda.sh
conda activate Andrew_env


cd /home/jdarne1/Eberhardt_Massive_Stars/SchroPo_public
python frozenHalo.py

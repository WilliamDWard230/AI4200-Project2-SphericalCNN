#! /bin/bash

#SBATCH --account=bs-fgdoane42
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/out.log
#SBATCH --error=logs/out.log

source ../dl_env/bin/activate

export PYTHONPATH="$PYTHONPATH:$(pwd)/lie_learn"
python -u train.py $1

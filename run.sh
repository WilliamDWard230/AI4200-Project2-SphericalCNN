#! /bin/bash

#SBATCH --account=bs-wdward43
#SBATCH --partition=batch-impulse
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/s10_out.log
#SBATCH --error=logs/s10_out.log

source ai4200-p1/bin/activate

python -u standard_train.py --train-fraction 0.10 2>&1 | while IFS= read -r line; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
done

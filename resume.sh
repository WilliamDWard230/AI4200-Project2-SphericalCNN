#!/bin/bash

#SBATCH --account=bs-wdward43
#SBATCH --partition=batch-warp
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/s_resume_out.log
#SBATCH --error=logs/s_resume_out.log

set -o pipefail
source ai4200-p1/bin/activate

python -u resume_spherical_100.py 2>&1 | while IFS= read -r line; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
done

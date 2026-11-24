
# Spherical CNNs

## Description

This project uses Spherical CNNs like the one proposed in https://arxiv.org/pdf/1801.10130 with $ S^2 $ and SO(3) convolution. Is uses a dataset of MNIST digits projected onto 3D spheres. The dataset, model architecture, and much of the code are from this original paper.


## How to Use

To run the code, first activate a virtual environment with Python 3.10 or newer installed. Then run the setup script:
```bash
source my_venv/bin/activate

chmod +x setup.sh
./setup.sh
```

Then set the constant parameters in the train.py files. For our models we used the following settings:
```python
NUM_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MNIST_PATH = "s2_mnist.gz"
SEED = 42
```

Now to run it, just run train.py with like on of the two following options:
```bash
# option to run with whole dataset
python train.py

# option to run with fraction of the dataset, 0.5 e.g.
python train.py --train-fraction=0.5
```

To run on the HPC with slurm, define the following run.sh file with the venv being used:

```bash
#! /bin/bash

#SBATCH --account=bs-wdward43
#SBATCH --partition=batch-warp
#SBATCH --cpus-per-task=12
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/s_out.log
#SBATCH --error=logs/s_out.log

source <venv>/bin/activate

python -u train.py 2>&1 | while IFS= read -r line; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"
done
```
and execute the file with slurm:
```bash
sbatch run.sh
```

A baseline can be trained identically, but the run script would need standard_train.py rather than train.py


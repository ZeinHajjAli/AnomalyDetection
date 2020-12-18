#!/bin/bash
#SBATCH --time=00:29:00
#SBATCH --mem-per-cpu=500G
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=result_DL_partial.out

module load python/3.8
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

pip install --no-index -r requirements.txt
python final_deeplearning_partial.py

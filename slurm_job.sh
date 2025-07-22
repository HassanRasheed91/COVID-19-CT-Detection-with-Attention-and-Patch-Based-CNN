# slurm_job.sh
#!/bin/bash
#SBATCH --job-name=covid_ct_job
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
module load anaconda/2023.3
source activate covid_ct_env
cd $SLURM_SUBMIT_DIR
python main.py
python eval.py

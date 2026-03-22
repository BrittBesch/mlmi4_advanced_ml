#!/bin/bash
#! Name of the job:
#SBATCH -J speech_kws
#! Which project should be charged:
#SBATCH -A MLMI-ttttd2-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#SBATCH --ntasks=1
#! Specify the number of GPUs per node:
#SBATCH --gres=gpu:1
#! How much wallclock time will be required?
#SBATCH --time=03:00:00
#! Automatic Slurm Logging (Creates these files in your logs folder):
#SBATCH --output=logs/speech_%j.out
#SBATCH --error=logs/speech_%j.err
#! Do not change:
#SBATCH -p ampere

#! set configurations to your REAL yaml file
options="configs/speech_config.yaml"

. /etc/profile.d/modules.sh                
module purge                               
module load rhel8/default-amp              
module load python/3.8.11/gcc-9.4.0-yb6rzr6   

source ${HOME}/.bashrc
source ${SLURM_SUBMIT_DIR}/mlmi4_env/bin/activate 

#! Full path to application executable: 
application="PYTHONPATH=. python -u src/training/train_fewshot_speech.py"

cd $SLURM_SUBMIT_DIR
mkdir -p ./data/speech
mkdir -p logs

echo -e "JobID: $SLURM_JOB_ID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

CMD="$application $options"
echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD
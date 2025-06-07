#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=hifigan-euskara # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=8-00:00 # No time limit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=/home/andoni.sudupe/.slurm/hifigan-euskara_spk.out.log
#SBATCH --error=/home/andoni.sudupe/.slurm/hifigan-euskara_spk.err.log
#SBATCH --chdir=/home/andoni.sudupe/speechbrain/recipes/Euskara/TTS/vocoder/hifigan_discrete

srun python ./train_spk.py ./hparams/train_spk.yaml --data_folder=/data/aholab/tts/eu/female/gaitu/
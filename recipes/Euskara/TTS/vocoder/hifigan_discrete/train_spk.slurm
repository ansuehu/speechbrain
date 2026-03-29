#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=hifigan-spk-2 # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=8-00:00 # No time limit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=/home/andoni.sudupe/speechbrain/.slurm/hifigan-euskara_spk_2.out.log
#SBATCH --error=/home/andoni.sudupe/speechbrain/.slurm/hifigan-euskara_spk_2.err.log
#SBATCH --chdir=/home/andoni.sudupe/speechbrain/

python ./recipes/Euskara/TTS/vocoder/hifigan_discrete/train_spk.py \
            ./recipes/Euskara/TTS/vocoder/hifigan_discrete/hparams/train_spk_sonora_2.yaml \
            --data_folder=/data/aholab/tts/eu/hifigan_spk/
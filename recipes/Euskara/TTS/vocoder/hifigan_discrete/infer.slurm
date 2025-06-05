#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=hifigan-infer # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=10:00:00 # No time limit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=/home/andoni.sudupe/.slurm/hifigan-infer.out.log
#SBATCH --error=/home/andoni.sudupe/.slurm/hifigan-infer.err.log
#SBATCH --chdir=/home/andoni.sudupe/speechbrain/recipes/Euskara/TTS/vocoder/hifigan_discrete

# Inference with a file or folder
# srun python -u ./infer.py \
#     --input_path path/to/audio_or_folder \
#     --hubert_repo Ansu/mHubert-basque-k1000-L9 \
#     --vocoder_repo Ansu/hifigan-hubert-k1000-miren

# Inference with a json file
srun python -u ./infer.py \
    --json_file ./results/hifi_gan/4322/save/test.json \
    --hubert_repo Ansu/mHubert-basque-k1000-L9 \
    --vocoder_repo Ansu/hifigan-hubert-k1000-miren \
    --output_dir ./test_results/ \
    --max_files 5
#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=hifigan-test # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=10:00:00 # No time limit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=/home/andoni.sudupe/.slurm/hifigan-test.out.log
#SBATCH --error=/home/andoni.sudupe/.slurm/hifigan-test.err.log
#SBATCH --chdir=/home/andoni.sudupe/speechbrain/recipes/Euskara/TTS/vocoder/hifigan_discrete

srun python -u ./test.py \
    --vocoder_repo ./results/hifigan_spk/4321/save/CKPT+2025-06-07+20-07-06+00 \
    --json ./results/hifigan_spk/4321/save/test.json \
    --codes_folder ./results/hifigan_spk/4321/save/codes \
    --output_file ./test_results/miren_nerea_300e.json \
    --spk_emb ./results/hifigan_spk/4321/save/speaker_embeddings \
    --batch_size 128

# srun python -u ./test.py \
#     --vocoder_repo ./results/hifi_gan/4322/save/CKPT+2025-06-05+10-13-16+00 \
#     --json ./results/hifi_gan/4322/save/test.json \
#     --codes_folder ./results/hifi_gan/4322/save/codes \
#     --output_file ./test_results/miren_400e.json \
#     --batch_size 128 \

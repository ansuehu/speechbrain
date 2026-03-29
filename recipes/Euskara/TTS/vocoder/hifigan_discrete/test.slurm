#!/usr/bin/env bash
#SBATCH --partition=gpu-H100
#SBATCH --job-name=hifigan-test # Name of the process
#SBATCH --gres=gpu:1 # Number of GPUs
#SBATCH --cpus-per-gpu=2 # Number of CPU cores (2 is reasonable)
#SBATCH --mem-per-gpu=16GB # RAM memory needed (8-16GB)
#SBATCH --time=2:00:00 # No time limit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=asudupe008@ikasle.ehu.eus
#SBATCH --output=/home/andoni.sudupe/speechbrain/.slurm/hifigan-test-sonora-2-900.out.log
#SBATCH --error=/home/andoni.sudupe/speechbrain/.slurm/hifigan-test-sonora-2-900.err.log
#SBATCH --chdir=/home/andoni.sudupe/speechbrain/

# spk_5
# srun python -u ./recipes/Euskara/TTS/vocoder/hifigan_discrete/test.py \
#     --vocoder_repo ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifigan_spk_5/4322/save/CKPT+2025-11-19+12-08-58+00 \
#     --json ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifigan_spk_5/4322/save/test.json \
#     --codes_folder ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifigan_spk_5/4322/save/codes \
#     --output_file ./recipes/Euskara/TTS/vocoder/hifigan_discrete/test_results_new/spk_5_200.json \
#     --speakers miren nerea jon marina alex \
#     --spk_emb ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifigan_spk_5/4322/save/speaker_embeddings \
#     --batch_size 128

# spk_sonora_2
srun python -u ./recipes/Euskara/TTS/vocoder/hifigan_discrete/test.py \
    --vocoder_repo ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifigan_spk_sonora_2/4321/save/CKPT+2025-11-22+08-35-35+00 \
    --json ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifigan_spk_sonora_2/4321/save/test.json \
    --codes_folder ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifigan_spk_sonora_2/4321/save/codes \
    --output_file ./recipes/Euskara/TTS/vocoder/hifigan_discrete/test_results_new/sonora_2_900.json \
    --speakers marina alex \
    --spk_emb ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifigan_spk_sonora_2/4321/save/speaker_embeddings \
    --batch_size 128

# gaitu_miren
# srun python -u ./recipes/Euskara/TTS/vocoder/hifigan_discrete/test.py \
#     --vocoder_repo ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifi_gan/4322/save/CKPT+2025-06-05+16-36-44+00 \
#     --json ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifi_gan/4322/save/test.json \
#     --codes_folder ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifi_gan/4322/save/codes \
#     --output_file ./recipes/Euskara/TTS/vocoder/hifigan_discrete/test_results_new/miren_900.json \
#     --batch_size 128 \

# spk_gaitu_2
# srun python -u ./recipes/Euskara/TTS/vocoder/hifigan_discrete/test.py \
#     --vocoder_repo ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifigan_gaitu_2/4321/save/CKPT+2025-06-09+12-19-04+00 \
#     --json ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifigan_gaitu_2/4321/save/test.json \
#     --codes_folder ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifigan_gaitu_2/4321/save/codes \
#     --output_file ./recipes/Euskara/TTS/vocoder/hifigan_discrete/test_results_new/gaitu_2_500.json \
#     --speakers miren nerea \
#     --spk_emb ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifigan_gaitu_2/4321/save/speaker_embeddings \
#     --batch_size 128

# spk_gaitu_3
# srun python -u ./recipes/Euskara/TTS/vocoder/hifigan_discrete/test.py \
#     --vocoder_repo ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifigan_gaitu_3/4322/save/CKPT+2025-06-13+18-27-39+00 \
#     --json ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifigan_gaitu_3/4322/save/test.json \
#     --codes_folder ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifigan_gaitu_3/4322/save/codes \
#     --output_file ./recipes/Euskara/TTS/vocoder/hifigan_discrete/test_results_new/gaitu_3_500.json \
#     --speakers miren nerea jon \
#     --spk_emb ./recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifigan_gaitu_3/4322/save/speaker_embeddings \
#     --batch_size 128



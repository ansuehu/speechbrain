import argparse
import torch
import torchaudio
from speechbrain.dataio.dataio import read_audio as sb_read_audio
import numpy as np
import os
import glob
import json
from transformers import Wav2Vec2Processor, HubertModel
import joblib
from speechbrain.inference.vocoders import UnitHIFIGAN
from huggingface_hub import hf_hub_download
import time
import random
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from python_speech_features import mfcc

def calculate_mcd(ref_audio, synth_audio, sr):
    ref_mfcc = mfcc(ref_audio.numpy(), samplerate=sr, numcep=13)
    synth_mfcc = mfcc(synth_audio.numpy(), samplerate=sr, numcep=13)
    distance, _ = fastdtw(ref_mfcc, synth_mfcc, dist=euclidean)
    mcd = (10.0 / np.log(10)) * distance / len(ref_mfcc)
    return mcd

def get_unique_filename(base_name, extension, directory="."):
    filename = f"{base_name}{extension}"
    counter = 1
    while os.path.exists(os.path.join(directory, filename)):
        filename = f"{base_name}_{counter}{extension}"
        counter += 1
    return os.path.join(directory, filename)

def generate_tokens(audio, processor, model, kmeans, device):
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs['input_values'] = inputs['input_values'].squeeze(0).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True).hidden_states[9]
    features = outputs.squeeze(0).cpu().numpy()
    tokens = kmeans.predict(features)
    tokens = torch.from_numpy(np.array(tokens))
    return tokens.unsqueeze(-1)

def read_audio(audio_path, sr):
    info = torchaudio.info(audio_path)
    audio = sb_read_audio(audio_path)
    audio = torchaudio.transforms.Resample(info.sample_rate, sr)(audio)
    return audio.unsqueeze(0)

def resolve_audio_files(input_path=None, json_file=None):
    audio_files = []
    if json_file:
        with open(json_file, "r") as f:
            data = json.load(f)
        entries = list(data.values())
        
        audio_files = [entry["wav"] for entry in entries if os.path.isfile(entry["wav"])]
    elif input_path:
        if os.path.isdir(input_path):
            exts = ["*.wav", "*.ogg", "*.mp3", "*.flac"]
            audio_files = []
            for ext in exts:
                audio_files.extend(glob.glob(os.path.join(input_path, ext)))
            audio_files.sort()
        elif os.path.isfile(input_path):
            audio_files = [input_path]
        else:
            raise ValueError(f"Invalid input path: {input_path}")
    else:
        raise ValueError("Must provide either --input_path or --json_file.")
    return audio_files

def main():
    parser = argparse.ArgumentParser(description="Tokenize and vocode audio using a HuBERT-based system.")
    parser.add_argument("--input_path", type=str, help="Path to an audio file or folder.")
    parser.add_argument("--json_file", type=str, help="Path to JSON file with audio filenames.")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to process from JSON.")
    parser.add_argument("--output_dir", type=str, default="./infer_results/", help="Directory to save output.")
    parser.add_argument("--hubert_repo", type=str, required=True, help="Hugging Face repo for HuBERT model.")
    parser.add_argument("--vocoder_repo", type=str, required=True, help="Hugging Face repo for vocoder.")
    parser.add_argument("--kmeans_path", type=str, default="kmeans/basque_hubert_k1000_L9.pt", help="Path to KMeans .pt file in Hubert repository.")
    parser.add_argument("--sr", type=int, default=16000, help="Target sampling rate.")
    parser.add_argument("--shuffle", type=bool, default=False, help="Shuffle audio list.")
    parser.add_argument("--spk", type=str, default=None, choices=[None, 'miren', 'nerea'], help="Speaker name for audio files.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    audio_files = resolve_audio_files(args.input_path, args.json_file)
    if not audio_files:
        print("No valid audio files found.")
        return
    print(f'Number of audios found: {len(audio_files)}')

    if args.max_files is not None:
        if len(audio_files) < args.max_files:
            print("Max files cannot be larger than found audios.")
            return

    if args.shuffle:
        random.shuffle(audio_files)

    processor = Wav2Vec2Processor.from_pretrained(args.hubert_repo)
    model = HubertModel.from_pretrained(args.hubert_repo).to(device).eval()
    hifi_gan_unit = UnitHIFIGAN.from_hparams(source=args.vocoder_repo, run_opts={"device":device}).eval()
    file_patterns = args.kmeans_path
    # kmeans_dir = snapshot_download(
    #         repo_id=args.hubert_repo, allow_patterns=file_patterns, cache_dir="./kmeans_folder"
    #     )
    file_path = hf_hub_download(repo_id=args.hubert_repo, filename=args.kmeans_path)

    kmeans = joblib.load(file_path)

    name_to_speaker = {
        'miren': 0,
        'nerea': 1,
    }
    if args.spk is not None:
        spk_emb = np.load(f"./results/hifigan_spk/4321/save/speaker_embeddings/{name_to_speaker[args.spk]}_ESP00001.npy")
        spk_emb = torch.FloatTensor(spk_emb).unsqueeze(0).to(device)
        print(f"Using speaker embedding for {args.spk} shape {spk_emb.shape}")


    for audio_path in audio_files[:args.max_files]:
        audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
        print(f"Processing {audio_path}...")

        audio = read_audio(audio_path, sr=args.sr)

        has = time.time()
        tokens = generate_tokens(audio, processor, model, kmeans, device)
        buk = time.time()
        if args.spk is not None:
            sig = hifi_gan_unit.decode_batch(tokens.unsqueeze(0).to(device), spk_emb)
        else:
            sig = hifi_gan_unit.decode_batch(tokens.unsqueeze(0).to(device))

        print(f'{buk-has:.2}s to encode and {time.time()-buk}s to decode')

        calculated_mcd = calculate_mcd(audio.squeeze(0).cpu(), sig.squeeze(0).cpu(), args.sr)
        print(f'MCD for {audio_basename}: {calculated_mcd:.2f}')
        output_path = get_unique_filename(audio_basename, ".wav", directory=args.output_dir)
        torchaudio.save(output_path, sig.squeeze(0).cpu(), args.sr)
        print(f'Audio saved in {output_path}')

if __name__ == "__main__":
    main()

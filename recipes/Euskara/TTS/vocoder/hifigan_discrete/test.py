import argparse
import json
import os
import numpy as np
import torchaudio
import torch
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from python_speech_features import mfcc
from importlib import import_module
from tqdm import tqdm
from speechbrain.inference.vocoders import UnitHIFIGAN
from speechbrain.dataio.dataio import read_audio as sb_read_audio
import time

def calculate_mcd(ref_audio, synth_audio, sr):
    ref_mfcc = mfcc(ref_audio.numpy(), samplerate=sr, numcep=13)
    synth_mfcc = mfcc(synth_audio.numpy(), samplerate=sr, numcep=13)
    distance, _ = fastdtw(ref_mfcc, synth_mfcc, dist=euclidean)
    mcd = (10.0 / np.log(10)) * distance / len(ref_mfcc)
    return mcd

def calculate_mcd_aligned(ref_audio, synth_audio, sr):
    ref_mfcc = mfcc(ref_audio.numpy(), samplerate=sr, numcep=13)
    synth_mfcc = mfcc(synth_audio.numpy(), samplerate=sr, numcep=13)

    min_len = min(len(ref_mfcc), len(synth_mfcc))
    ref_mfcc = ref_mfcc[:min_len]
    synth_mfcc = synth_mfcc[:min_len]

    diff = ref_mfcc - synth_mfcc
    dist = np.sqrt((diff ** 2).sum(axis=1))
    mcd = (10.0 / np.log(10)) * np.mean(dist)
    return mcd

def calculate_mcd_tensor_batch(ref_wavs, synth_wavs, sample_rate):
    # ref_wavs, synth_wavs: (B, T)
    assert ref_wavs.shape == synth_wavs.shape, "Ref and Synth shapes must match"

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=13,
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
    )

    # Compute MFCCs
    ref_mfcc = mfcc_transform(ref_wavs)       # (B, 13, T')
    synth_mfcc = mfcc_transform(synth_wavs)   # (B, 13, T')

    # Truncate to same length
    min_len = min(ref_mfcc.shape[-1], synth_mfcc.shape[-1])
    ref_mfcc = ref_mfcc[:, :, :min_len]
    synth_mfcc = synth_mfcc[:, :, :min_len]

    # Compute Euclidean distance per frame
    diff = ref_mfcc - synth_mfcc              # (B, 13, T)
    dist = torch.norm(diff, dim=1)            # (B, T)
    mcd = (10.0 / torch.log(torch.tensor(10.0))) * torch.mean(dist, dim=1)  # (B,)

    return mcd.mean().item()


def read_audio(audio_path, sr):
    info = torchaudio.info(audio_path)
    audio = sb_read_audio(audio_path)
    audio = torchaudio.transforms.Resample(info.sample_rate, sr)(audio)
    return audio

def pad_and_stack(tensors, pad_value=0):
    lengths = [t.shape[0] for t in tensors]
    max_len = max(lengths)
    dim = tensors[0].shape[1]
    padded = torch.full((len(tensors), max_len, dim), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        padded[i, :t.shape[0]] = t
    return padded

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vocoder Evaluation Script using MCD")
    parser.add_argument('--vocoder_repo', type=str, required=True, help='Name of the vocoder repository (importable module)')
    parser.add_argument('--json', type=str, required=True, help='Path to JSON file with wav metadata')
    parser.add_argument('--codes_folder', type=str, required=True, help='Folder containing .npy code files')
    parser.add_argument('--output_file', type=str, default='mcd_results.json', help='Output file to save results')
    parser.add_argument('--spk_emb', type=str, default=None, help='Speaker embedding folder')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate for processing')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hifi_gan_unit = UnitHIFIGAN.from_hparams(source=args.vocoder_repo, run_opts={"device":device}).eval()

    with open(args.json, 'r') as f:
        metadata = json.load(f)

    batch_keys = []
    batch_codes = []
    batch_waveforms = []
    if args.spk_emb is not None:
        batch_spk_emb = []
    mcd_scores = {}

    keys = list(metadata.keys())
    print(f"Processing {len(keys)} files with batch size {args.batch_size}")
    for i in tqdm(range(0, len(keys), args.batch_size)):
        batch = keys[i:i + args.batch_size]
        batch_keys = []
        batch_codes = []
        batch_waveforms = []
        if args.spk_emb is not None:
            batch_spk_emb = []

        hasi = time.time()
        for key in batch:
            wav_path = metadata[key]['wav']
            waveform = read_audio(wav_path, args.sr)
            code_path = os.path.join(args.codes_folder, f"{key}.npy")
            codes = np.load(code_path)

            if args.spk_emb is not None:
                spk_path = os.path.join(args.spk_emb, f"{key}.npy")
                spk_emb = np.load(spk_path)
                batch_spk_emb.append(torch.tensor(spk_emb).to(device))

            batch_codes.append(torch.tensor(codes).to(device))
            batch_waveforms.append(waveform)
            batch_keys.append(key)
        # print(f"Batch {i // args.batch_size + 1} processed in {time.time() - hasi:.2f} seconds")
        # Stack into batch and move to GPU
        # code_batch = torch.stack(batch_codes).to(device)
        # code_batch = [code.to(device) for code in batch_codes]
        hasi = time.time()
        padded_codes = pad_and_stack(batch_codes)
        # print(f"Padding and stacking codes took {time.time() - hasi:.2f} seconds")

        if args.spk_emb is not None:
            batch_spk_emb = torch.stack(batch_spk_emb).to(device)
            # batch_spk_emb = [spk.to(device) for spk in batch_spk_emb]

        hasi = time.time()
        with torch.no_grad():
            if args.spk_emb is not None:
                synth_batch = hifi_gan_unit.decode_batch(padded_codes, spk=batch_spk_emb)
            else:
                synth_batch = hifi_gan_unit.decode_batch(padded_codes)

        # print(f"Decoding batch took {time.time() - hasi:.2f} seconds")

        hasi = time.time()
        for k, ref_wave, synth_wave in zip(batch_keys, batch_waveforms, synth_batch.cpu()):
            min_len = min(ref_wave.shape[-1], synth_wave.shape[-1])
            # mcd = calculate_mcd(ref_wave[..., :min_len].squeeze(), synth_wave[..., :min_len].squeeze(), args.sr)
            mcd = calculate_mcd_aligned(ref_wave[..., :min_len].squeeze(), synth_wave[..., :min_len].squeeze(), args.sr)
            # mcd = calculate_mcd_tensor_batch(ref_wave[..., :min_len].squeeze(), synth_wave[..., :min_len].squeeze(), args.sr)
            mcd_scores[k] = mcd
        print(f"Calculating MCD for batch took {time.time() - hasi:.2f} seconds")

    mean_mcd = np.mean(list(mcd_scores.values()))
    results = {"mean_mcd": mean_mcd,"individual_mcd": mcd_scores}
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Mean MCD: {mean_mcd:.4f}")
import argparse
import json
import os
import numpy as np
import torchaudio
import torch
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from python_speech_features import mfcc
from tqdm import tqdm
from speechbrain.inference.vocoders import UnitHIFIGAN
from speechbrain.dataio.dataio import read_audio as sb_read_audio
import time

PRINT_LEN = True

import librosa
import math
import numpy as np
import pyworld
import pysptk
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


# ================================================= #
# calculate the Mel-Cepstral Distortion (MCD) value #
# ================================================= #
class Calculate_MCD(object):
	"""docstring for Calculate_MCD"""
	def __init__(self, MCD_mode):
		super(Calculate_MCD, self).__init__()
		# self.args = args
		self.MCD_mode = MCD_mode
		self.SAMPLING_RATE = 22050
		self.FRAME_PERIOD = 5.0
		self.log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0) # 6.141851463713754
	
	def load_wav(self, wav_file, sample_rate):
		"""
		Load a wav file with librosa.
		:param wav_file: path to wav file
		:param sr: sampling rate
		:return: audio time series numpy array
		"""
		wav, _ = librosa.load(wav_file, sr=sample_rate, mono=True)
		return wav

	# distance metric
	def log_spec_dB_dist(self, x, y):
		# log_spec_dB_const = 10.0 / math.log(10.0) * math.sqrt(2.0)
		diff = x - y
		return self.log_spec_dB_const * math.sqrt(np.inner(diff, diff))

	# calculate distance (metric)
	# def calculate_mcd_distance(self, x, y, distance, path):
	def calculate_mcd_distance(self, x, y, path):
		'''
		param path: pairs between x and y
		'''
		pathx = list(map(lambda l: l[0], path))
		pathy = list(map(lambda l: l[1], path))
		x, y = x[pathx], y[pathy]
		frames_tot = x.shape[0]       # length of pairs

		z = x - y
		min_cost_tot = np.sqrt((z * z).sum(-1)).sum()

		return frames_tot, min_cost_tot

	# extract acoustic features
	# alpha = 0.65  # commonly used at 22050 Hz
	def wav2mcep_numpy(self, loaded_wav, alpha=0.65, fft_size=512):
        
		# Use WORLD vocoder to spectral envelope
		_, sp, _ = pyworld.wav2world(loaded_wav.astype(np.double), fs=self.SAMPLING_RATE,
									   frame_period=self.FRAME_PERIOD, fft_size=fft_size)

		# Extract MCEP features
		mcep = pysptk.sptk.mcep(sp, order=13, alpha=alpha, maxiter=0,
							   etype=1, eps=1.0E-8, min_det=0.0, itype=3)

		return mcep

	# calculate the Mel-Cepstral Distortion (MCD) value
	def average_mcd(self, loaded_ref_wav, loaded_syn_wav, cost_function, MCD_mode):
		"""
		Calculate the average MCD.
		:param ref_mcep_files: list of strings, paths to MCEP target reference files
		:param synth_mcep_files: list of strings, paths to MCEP converted synthesised files
		:param cost_function: distance metric used
		:param plain: if plain=True, use Dynamic Time Warping (dtw)
		:returns: average MCD, total frames processed
		"""
		# load wav from given wav file
		# loaded_ref_wav = self.load_wav(ref_audio_file, sample_rate=self.SAMPLING_RATE)
		# loaded_syn_wav = self.load_wav(syn_audio_file, sample_rate=self.SAMPLING_RATE)

		if MCD_mode == "plain":
			# pad 0
			if len(loaded_ref_wav)<len(loaded_syn_wav):
				loaded_ref_wav = np.pad(loaded_ref_wav, (0, len(loaded_syn_wav)-len(loaded_ref_wav)))
			else:
				loaded_syn_wav = np.pad(loaded_syn_wav, (0, len(loaded_ref_wav)-len(loaded_syn_wav)))

		# extract MCEP features (vectors): 2D matrix (num x mcep_size)
		ref_mcep_vec = self.wav2mcep_numpy(loaded_ref_wav)
		syn_mcep_vec = self.wav2mcep_numpy(loaded_syn_wav)

		if MCD_mode == "plain":
			# print("Calculate plain MCD ...")
			path = []
			# for i in range(num_temp):
			for i in range(len(ref_mcep_vec)):
				path.append((i, i))
		elif MCD_mode == "dtw":
			# print("Calculate MCD-dtw ...")
			_, path = fastdtw(ref_mcep_vec[:, 1:], syn_mcep_vec[:, 1:], dist=euclidean)
		elif MCD_mode == "dtw_sl":
			# print("Calculate MCD-dtw-sl ...")
			cof = len(ref_mcep_vec)/len(syn_mcep_vec) if len(ref_mcep_vec)>len(syn_mcep_vec) else len(syn_mcep_vec)/len(ref_mcep_vec)
			_, path = fastdtw(ref_mcep_vec[:, 1:], syn_mcep_vec[:, 1:], dist=euclidean)

		frames_tot, min_cost_tot = self.calculate_mcd_distance(ref_mcep_vec, syn_mcep_vec, path)

		if MCD_mode == "dtw_sl":
			mean_mcd = cof * self.log_spec_dB_const * min_cost_tot / frames_tot
		else:
			mean_mcd = self.log_spec_dB_const * min_cost_tot / frames_tot

		return mean_mcd

	# calculate mcd
	def calculate_mcd(self, reference_audio, synthesized_audio):
		# extract acoustic features
		mean_mcd = self.average_mcd(reference_audio, synthesized_audio, self.log_spec_dB_dist, self.MCD_mode)

		return mean_mcd

def pad_and_stack(tensors, pad_value=0):
    lengths = [t.shape[0] for t in tensors]
    max_len = max(lengths)
    dim = tensors[0].shape[1]
    padded = torch.full((len(tensors), max_len, dim), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        padded[i, :t.shape[0]] = t
    return padded

def read_audio(audio_path, sr):
    info = torchaudio.info(audio_path)
    audio = sb_read_audio(audio_path)
    audio = torchaudio.transforms.Resample(info.sample_rate, sr)(audio)
    return audio

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vocoder Evaluation Script using MCD")
    parser.add_argument('--vocoder_repo', type=str, required=True, help='Name of the vocoder repository (importable module)')
    parser.add_argument('--json', type=str, required=True, help='Path to JSON file with wav metadata')
    parser.add_argument('--codes_folder', type=str, required=True, help='Folder containing .npy code files')
    parser.add_argument('--output_file', type=str, default='mcd_results.json', help='Output file to save results')
    parser.add_argument('--speakers', type=str, default=None, nargs='+', help='Comma-separated list of speakers to process')
    parser.add_argument('--spk_emb', type=str, default=None, help='Speaker embedding folder')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing')
    parser.add_argument('--sr', type=int, default=16000, help='Sample rate for processing')
    args = parser.parse_args()
    
    mcd_toolbox = Calculate_MCD(MCD_mode="plain")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hifi_gan_unit = UnitHIFIGAN.from_hparams(source=args.vocoder_repo, run_opts={"device":device}).eval()

    with open(args.json, 'r') as f:
        metadata = json.load(f)

    batch_keys = []
    batch_codes = []
    batch_waveforms = []
    mcd_scores = {}
    if args.speakers is not None:
        mcd_speaker = {}
        id2speaker = {}
        batch_spk_emb = []
        for i, speaker in enumerate(args.speakers):
            id2speaker[i] = speaker
            mcd_speaker[speaker] = []

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
        for j, (k, ref_wave, synth_wave) in enumerate(zip(batch_keys, batch_waveforms, synth_batch.cpu())):
            min_len = min(ref_wave.shape[-1], synth_wave.shape[-1])
            # mcd = calculate_mcd_aligned(ref_wave[..., :min_len].squeeze(), synth_wave[..., :min_len].squeeze(), args.sr)
            mcd = mcd_toolbox.calculate_mcd(ref_wave[..., :min_len].squeeze().numpy(), synth_wave[..., :min_len].squeeze().numpy())
            mcd_scores[k] = mcd

            if i == 0 and j == 0:
                output_prefix = f".{args.output_file.split('.')[1]}"
                print(f"Saving first synthesized and reference audio to {output_prefix}_synth.wav and {output_prefix}_ref.wav")
                torchaudio.save(f"{output_prefix}_synth.wav", synth_wave[..., :min_len], args.sr)
                torchaudio.save(f"{output_prefix}_ref.wav", ref_wave[..., :min_len].unsqueeze(0), args.sr)
                print(f"Saved first synthesized and reference audio to {output_prefix}_synth.wav and {output_prefix}_ref.wav")

            if args.speakers is not None:
                id = int(k.split('_')[0])
                mcd_speaker[id2speaker[id]].append(mcd)
                
    mean_mcd = np.mean(list(mcd_scores.values()))
    if args.speakers is not None:
        for speaker, scores in mcd_speaker.items():
            mcd_speaker[speaker] = np.mean(scores)

    # Save results and args
    results = {"args": vars(args), "mean_mcd": mean_mcd}
    results['mcd_speaker'] = mcd_speaker if args.speakers is not None else None
    results['mcd_scores'] = mcd_scores

    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Mean MCD: {mean_mcd:.4f}")
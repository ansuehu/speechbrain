import torch
from speechbrain.inference.vocoders import UnitHIFIGAN
from transformers import Wav2Vec2Processor, HubertModel
import joblib
import torchaudio
import librosa
import numpy as np
import speechbrain as sb
# import soundfile as sf

hifi_gan_unit = UnitHIFIGAN.from_hparams(source="/home/andoni.sudupe/speechbrain/recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifi_gan/4322/save/CKPT+2025-06-03+14-38-39+00")
# tokens = torch.randint(0, 99, (100, 1))
# audio_basename = "random"

model_name = "Ansu/mHubert-basque-k1000-L9"
# model_name = "Ansu/mHubert-basque-ASR"
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = HubertModel.from_pretrained(model_name)
model.eval()
kmeans = joblib.load("/home/andoni.sudupe/mHubert_finetune/checkpoints/kmeans/basque_hubert_k1000_L9.pkl")

audio_path = "./audio_2025-06-03_14-48-41.ogg"
# audio_path = "/data/aholab/tts/eu/female/gaitu/miren-hifigan/wavs/ESP00001.wav"
audio_basename = audio_path.split("/")[-1].split(".")[0]

# info = torchaudio.info(audio_path)
# audio, _ = librosa.load(audio_path, sr=16000)

info = torchaudio.info(audio_path)
audio = sb.dataio.dataio.read_audio(audio_path)
audio = torchaudio.transforms.Resample(
    info.sample_rate,
    16000,
)(audio)
audio = audio.unsqueeze(0)

inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
inputs['input_values'] = inputs['input_values'].squeeze(0)
print(inputs['input_values'].shape)
with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True).hidden_states[9]
features = outputs.squeeze(0).cpu().numpy()
# print(features[0])
tokens = kmeans.predict(features)
print(tokens)
tokens = torch.from_numpy(np.array(tokens))
tokens = tokens.unsqueeze(-1)
# # print(tokens)

# tokens = np.load('/home/andoni.sudupe/speechbrain/recipes/Euskara/TTS/vocoder/hifigan_discrete/results/hifi_gan/4322/save/codes/ESP00001.npy')
# audio_basename = "ESP00001"
# tokens = torch.from_numpy(tokens)
# tokens = tokens.unsqueeze(-1)
print(tokens.shape)
waveform = hifi_gan_unit.decode_unit(tokens)
# Save the waveform to a file
output_path = f"/home/andoni.sudupe/speechbrain/recipes/Euskara/TTS/vocoder/hifigan_discrete/infer_results/{audio_basename}.wav"
torchaudio.save(output_path, waveform, 16000)



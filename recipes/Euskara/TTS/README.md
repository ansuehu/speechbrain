# Text-to-Speech (with LJSpeech)
This folder contains the recipes for training TTS systems (including vocoders) with the popular LJSpeech dataset.

# Dataset
The dataset can be downloaded from here:
https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2

# Installing Extra Dependencies

Before proceeding, ensure you have installed the necessary additional dependencies. To do this, simply run the following command in your terminal:

```
pip install -r extra_requirements.txt
```

# HiFiGAN (Vocoder)
The subfolder "vocoder/hifigan/" contains the [HiFiGAN vocoder](https://arxiv.org/pdf/2010.05646.pdf).
The vocoder is a neural network that converts a spectrogram into a waveform (it can be used on top of Tacotron2/FastSpeech2).

We suggest using `tensorboard_logger` by setting `use_tensorboard: True` in the yaml file, thus `Tensorboard` should be installed.

To run this recipe, go into the "vocoder/hifigan/" folder and run:

```
python train.py hparams/train.yaml --data_folder /path/to/LJspeech
```

Training takes about 10 minutes/epoch on an nvidia RTX8000.

The training logs are available [here](https://www.dropbox.com/sh/m2xrdssiroipn8g/AAD-TqPYLrSg6eNxUkcImeg4a?dl=0)

You can find the pre-trained model with an easy-inference function on [HuggingFace](https://huggingface.co/speechbrain/tts-hifigan-ljspeech).


# HiFiGAN Unit Vocoder
The subfolder "vocoder/hifigan_discrete/" contains the [HiFiGAN Unit vocoder](https://arxiv.org/abs/2406.10735). This vocoder is a neural network designed to transform discrete self-supervised representations into waveform data.
This is suitable for a wide range of generative tasks such as speech enhancement, separation, text-to-speech, voice cloning, etc. Please read [DASB - Discrete Audio and Speech Benchmark](https://arxiv.org/abs/2406.14294) for more information.

To run this recipe successfully, start by installing the necessary extra dependencies:

```bash
pip install -r extra_requirements.txt
```

Before training the vocoder, you need to choose a speech encoder to extract representations that will be used as discrete audio input. We support k-means models using features from HuBERT, WavLM, or Wav2Vec2. Below are the available self-supervised speech encoders for which we provide pre-trained k-means checkpoints:

| Encoder  | HF model                                |
|----------|-----------------------------------------|
| HuBERT   | facebook/hubert-large-ll60k             |
| Wav2Vec2 | facebook/wav2vec2-large-960h-lv60-self  |
| WavLM    | microsoft/wavlm-large                   |

Checkpoints are available in the HF [SSL_Quantization](https://huggingface.co/speechbrain/SSL_Quantization) repository. Alternatively, you can train your own k-means model by following instructions in the "LJSpeech/quantization" README.

Next, configure the SSL model type, k-means model, and corresponding hub in your YAML configuration file. Follow these steps:

1. Navigate to the "vocoder/hifigan_discrete/hparams" folder and open "train.yaml" file.
2. Modify the `encoder_type` field to specify one of the SSL models: "HuBERT", "WavLM", or "Wav2Vec2".
3. Update the `encoder_hub` field with the specific name of the SSL Hub associated with your chosen model type.

If you have trained your own k-means model, follow these additional steps:

4. Update the `kmeans_folder` field with the specific name of the SSL Hub containing your trained k-means model. Please follow the same file structure as the official one in [SSL_Quantization](https://huggingface.co/speechbrain/SSL_Quantization).
5. Update the `kmeans_dataset` field with the specific name of the dataset on which the k-means model was trained.
6. Update the `num_clusters` field according to the number of clusters of your k-means model.

Finally, navigate back to the "vocoder/hifigan_discrete/" folder and run the following command:

```bash
python train.py hparams/train.yaml --data_folder=/path/to/LJspeech
```

Training typically takes around 4 minutes per epoch when using an NVIDIA A100 40G.


# **About SpeechBrain**
- Website: https://speechbrain.github.io/
- Code: https://github.com/speechbrain/speechbrain/
- HuggingFace: https://huggingface.co/speechbrain/


# **Citing SpeechBrain**
Please, cite SpeechBrain if you use it for your research or business.

```bibtex
@misc{speechbrainV1,
  title={Open-Source Conversational AI with SpeechBrain 1.0},
  author={Mirco Ravanelli and Titouan Parcollet and Adel Moumen and Sylvain de Langen and Cem Subakan and Peter Plantinga and Yingzhi Wang and Pooneh Mousavi and Luca Della Libera and Artem Ploujnikov and Francesco Paissan and Davide Borra and Salah Zaiem and Zeyu Zhao and Shucong Zhang and Georgios Karakasidis and Sung-Lin Yeh and Pierre Champion and Aku Rouhe and Rudolf Braun and Florian Mai and Juan Zuluaga-Gomez and Seyed Mahed Mousavi and Andreas Nautsch and Xuechen Liu and Sangeet Sagar and Jarod Duret and Salima Mdhaffar and Gaelle Laperriere and Mickael Rouvier and Renato De Mori and Yannick Esteve},
  year={2024},
  eprint={2407.00463},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2407.00463},
}
@misc{speechbrain,
  title={{SpeechBrain}: A General-Purpose Speech Toolkit},
  author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio},
  year={2021},
  eprint={2106.04624},
  archivePrefix={arXiv},
  primaryClass={eess.AS},
  note={arXiv:2106.04624}
}
```


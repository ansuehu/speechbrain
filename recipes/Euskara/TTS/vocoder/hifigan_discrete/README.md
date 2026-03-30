# Text-to-Speech (with Basque speakers)
This folder contains the recipe for training a Unit-HiFiGAN with basque speakers. THe hole folder is based on `speechbrain/recipes/LibriTTS/vocoder/hifigan_discrete`, go there if you want to see the original implementation.

# Dataset
The dataset structure is simple: 1 folder per speaker and all the wav files inside that folder. Then we will select which speaker we want to use.

# Training

Before training the vocoder, you need to choose a speech encoder to extract representations that will be used as discrete audio input. I added support for using the HuggingFace module of HuBERT, and a k-means for Euskera.

Checkpoints are available in the HF [mHuBERT-eus](https://huggingface.co/Ansu/mHubert-basque-k1000-L9) repository.

All the configuration is made in the YAML file. There are 2 types of YAML; one for single-speaker configuration and the other for multi-speaker. I saw that using more than 1 speaker can boost performance of the model, so I will follow that. 

The important things are the following:

1. Experiment Parameters and setup

Here we can change the save folder and the max epochs for the training.

2. Data files and pre-processing

Here we specify the data we are going to use. First of all, we need to select the speakers in the `speaker_subsets`. 

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


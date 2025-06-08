"""
LibriTTS data preparation

Authors
 * Pradnya Kandarkar 2022
"""

import json
import os
import random

import torch
import torchaudio
from tqdm import tqdm

from speechbrain.inference.text import GraphemeToPhoneme
from speechbrain.utils.data_utils import get_all_files
from speechbrain.utils.logger import get_logger
from speechbrain.utils.text_to_sequence import _g2p_keep_punctuations

logger = get_logger(__name__)
LIBRITTS_URL_PREFIX = "https://www.openslr.org/resources/60/"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_multi_spk(
    data_folder,
    save_json_train,
    save_json_valid,
    save_json_test,
    sample_rate,
    split_ratio=[80, 10, 10],
    speaker_subsets=None,
    train_split=None,
    valid_split=None,
    test_split=None,
    seed=1234,
    model_name=None,
    skip_prep=False,
):
    """
    Prepares the json files for the LibriTTS dataset.
    Downloads the dataset if it is not found in the `data_folder` as expected.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the LibriTTS dataset is stored.
    save_json_train : str
        Path where the train data specification file will be saved.
    save_json_valid : str
        Path where the validation data specification file will be saved.
    save_json_test : str
        Path where the test data specification file will be saved.
    sample_rate : int
        The sample rate to be used for the dataset
    split_ratio : list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.
    libritts_subsets: list
        List of librispeech subsets to use (e.g., dev-clean, train-clean-100, ...) for the experiment.
        This parameter will be ignored if explicit data splits are provided.
        Explicit data splits parameters: "train_split", "valid_split", "test_split"
    train_split : list
        List of librispeech subsets to use (e.g.,train-clean-100, train-clean-360) for the experiment training stage.
    valid_split : list
        List of librispeech subsets to use (e.g., dev-clean) for the experiment validation stage.
    test_split : list
        List of librispeech subsets to use (e.g., test-clean) for the experiment testing stage.
    seed : int
        Seed value
    model_name : str
        Model name (used to prepare additional model specific data)
    skip_prep: Bool
        If True, skip preparation.

    Returns
    -------
    None
    """

    if skip_prep:
        return

    # Setting the seed value
    random.seed(seed)

    # Checks if this phase is already done (if so, skips it)
    if skip(save_json_train, save_json_valid, save_json_test):
        logger.info("Preparation completed in previous run, skipping.")
        return

    logger.info(
        f"Creating {save_json_train}, {save_json_valid}, and {save_json_test}"
    )

    # Creates data manifest files according to the data splits
    wav_list = prepare_split(data_folder, speaker_subsets)

    # Random split the signal list into train, valid, and test sets.
    data_split = split_sets(wav_list, split_ratio)
    # Creating json files
    create_json(
        data_split["train"], save_json_train, sample_rate, model_name, speaker_subsets
    )
    create_json(
        data_split["valid"], save_json_valid, sample_rate, model_name, speaker_subsets
    )
    create_json(data_split["test"], save_json_test, sample_rate, model_name, speaker_subsets)


def prepare_split(data_folder, speaker_subsets):
    """
    Processes the provided list of LibriTTS subsets and creates a list of all the .wav files present in the subsets.
    Downloads the LibriTTS subsets as required.

    Arguments
    ---------
    data_folder : str
        Path to the folder where the LibriTTS dataset is stored
    split_list : list
        List of librispeech subsets to process (e.g., dev-clean, train-clean-100, ...)

    Returns
    -------
    wav_list : list
        List of all .wav files to be processed
    """
    extension = [".wav"]  # The expected extension for audio files
    exclude = ["_24k"]
    wav_list = list()  # Stores all audio file paths for the dataset

    # For every subset of the dataset, if it doesn't exist, downloads it
    for subset_name in speaker_subsets:
        subset_folder = os.path.join(data_folder, subset_name)
        logger.info(f'Looking in {subset_folder}')
        if not check_folders(subset_folder):
            logger.info(
                f"No data found for {subset_name}. Checking for an archive file."
            )
            quit()
        # Collects all files matching the provided extension
        split_files = get_all_files(subset_folder, match_and=extension, exclude_or=exclude)
        wav_list.extend(split_files)
        logger.info(f'Number of files: {len(split_files)}')
    return wav_list


def create_json(wav_list, json_file, sample_rate, model_name=None, speaker_subsets=None):
    """
    Creates the json file given a list of wav files.
    Arguments
    ---------
    wav_list : list of str
        The list of wav files.
    json_file : str
        The path of the output json file
    sample_rate : int
        The sample rate to be used for the dataset
    model_name : str
        Model name (used to prepare additional model specific data)
    """

    json_dict = {}
    name_to_id = {k:i for i, k in enumerate(speaker_subsets)}

    # Processes all the wav files in the list
    for wav_file in tqdm(wav_list):
        if not wav_file.endswith('.wav') or '_24k' in wav_file:
            continue
        # Reads the signal
        signal, sig_sr = torchaudio.load(wav_file)
        duration = signal.shape[1] / sig_sr

        # TODO add better way to filter short utterances
        if duration < 1.0:
            continue

        # Manipulates path to get relative path and uttid
        path_parts = wav_file.split(os.path.sep)
        uttid, _ = os.path.splitext(path_parts[-1])
        # relative_path = os.path.join("{data_root}", *path_parts[-4:])

        # Gets the path for the text files and extracts the input text
        normalized_text_path = os.path.join(
            "/", *path_parts[:-1], uttid + ".txt"
        )
        try:
            with open(normalized_text_path, encoding="utf-8") as f:
                normalized_text = f.read()
                if normalized_text.__contains__("{"):
                    normalized_text = normalized_text.replace("{", "")
                if normalized_text.__contains__("}"):
                    normalized_text = normalized_text.replace("}", "")
        except FileNotFoundError:
            normalized_text = " "
            print(f"Warning: The file {normalized_text_path} does not exist.")
            # continue

        # Resamples the audio file if required
        # if sig_sr != sample_rate:
        #     resampled_signal = torchaudio.functional.resample(
        #         signal, sig_sr, sample_rate
        #     )
        #     os.unlink(wav_file)
        #     torchaudio.save(wav_file, resampled_signal, sample_rate=sample_rate)

        # Gets the speaker-id from the utterance-id
        spk_id = name_to_id[path_parts[-2]]

        # Creates an entry for the utterance
        key = f'{spk_id}_{uttid}'
        json_dict[key] = {
            "uttid": uttid,
            "wav": wav_file,
            "duration": duration,
            "spk_id": spk_id,
            "label": normalized_text,
            "segment": True if "train" in json_file else False,
        }

    # Writes the dictionary to the json file
    with open(json_file, mode="w", encoding="utf-8") as json_f:
        json.dump(json_dict, json_f, indent=2)

    logger.info(f"{json_file} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Arguments
    ---------
    *filenames : tuple
        Set of filenames to check for existence.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def split_sets(wav_list, split_ratio):
    """Randomly splits the wav list into training, validation, and test lists.

    Arguments
    ---------
    wav_list : list
        list of all the signals in the dataset
    split_ratio: list
        List composed of three integers that sets split ratios for train, valid,
        and test sets, respectively. For instance split_ratio=[80, 10, 10] will
        assign 80% of the sentences to training, 10% for validation, and 10%
        for test.

    Returns
    -------
    dictionary containing train, valid, and test splits.
    """
    # Random shuffles the list
    random.shuffle(wav_list)
    tot_split = sum(split_ratio)
    tot_snts = len(wav_list)
    data_split = {}
    splits = ["train", "valid"]

    for i, split in enumerate(splits):
        n_snts = int(tot_snts * split_ratio[i] / tot_split)
        data_split[split] = wav_list[0:n_snts]
        del wav_list[0:n_snts]
    data_split["test"] = wav_list

    return data_split


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True

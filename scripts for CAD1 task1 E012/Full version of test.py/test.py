""" Run the baseline enhancement. """
from __future__ import annotations

# pylint: disable=import-error
import json
import logging
import shutil
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy.io import wavfile
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from recipes.cad1.task1.baseline.enhance import (
    decompose_signal,
    get_device,
    process_stems_for_listener,
    save_flac_signal,
)

from recipes.cad1.task1.baseline.evaluate import make_song_listener_list
from multibandCompressor import MultibandCompressor

# pylint: disable=too-many-locals

logger = logging.getLogger(__name__)


def pack_submission(
    team_id: str,
    root_dir: str | Path,
    base_dir: str | Path = ".",
) -> None:
    """
    Pack the submission files into an archive file.

    Args:
        team_id (str): Team ID.
        root_dir (str | Path): Root directory of the archived file.
        base_dir (str | Path): Base directory to archive. Defaults to ".".
    """
    # Pack the submission files
    logger.info(f"Packing submission files for team {team_id}...")
    shutil.make_archive(
        f"submission_{team_id}",
        "zip",
        root_dir=root_dir,
        base_dir=base_dir,
    )

def change_volume(audio, db):
    factor = np.power(10.0, db / 20.0)
    print(f"The audio should be multiplied by a factor of {factor}")
    return audio * factor

def judge_volume(str1,audio,db1,db2,db3,db4):
    if str1 == "left_vocals" or str1 == "right_vocals":
        return change_volume(audio,db1)
    elif str1 == "left_bass" or str1 == "right_bass":
        return change_volume(audio,db2)
    elif str1 == "left_drums" or str1 == "right_drums":
        return change_volume(audio,db3)
    elif str1 == "left_other" or str1 == "right_other":
        return change_volume(audio,db4)  

def classify_frequency_listeners(audiogram):
    # 0(None): <20 ,  1(mild): 20-34,  2(moderate): 35-49, 
    # 3(moderately severe):50-64, 4(severe): >65

    result = list([0]*audiogram)

    for index in range(audiogram.shape[0]):

        if audiogram[index] < 20:
            result[index] = 0
        elif audiogram[index] >= 20 and audiogram[index] < 35:
            result[index] = 1
        elif audiogram[index] >= 35 and audiogram[index] < 50:
            result[index] = 2
        elif audiogram[index] >= 50 and audiogram[index] < 65:
            result[index] = 3
        else:
            result[index] = 4
    
    return result

def analyze_audiogram(audiogram_level):
    advanced_remixing = True
    db2 = 0
    db3 = 0
    db4 = 0
    if max(audiogram_level) <= 2:
        advanced_remixing = False

    # bass
    if audiogram_level[0] >= 3:
        db2 = -0.5
    # other 
    if max(audiogram_level[1:6]) >= 3:
        db4 = -0.5
    # drums
    if max(audiogram_level[6:8]) >= 3:
        db3 = -0.5

    return advanced_remixing, db2, db3, db4
    
    
def selectbands(audio_array,samplerate):
    freq_set = [500, 1000, 2000, 3000, 4000, 6000, 8000]
    fft_out = np.fft.rfft(audio_array)
    freqs = np.fft.rfftfreq(len(audio_array), 1 / samplerate)
    amplitudes = np.abs(fft_out)
    max_amplitude = np.max(amplitudes)
    threshold = 1 / 4 * max_amplitude
    selected_freq_index = np.where(amplitudes > threshold)
    selected_freq = freqs[selected_freq_index]
    count = np.array([0, 0, 0, 0, 0, 0])
    bands = []
    for i in range(len(freq_set)-1):
        for j in range(selected_freq.shape[0]):
            if selected_freq[j] > freq_set[i] and selected_freq[j]<=freq_set[i+1]:
                count[i]+=1
    indices = np.argpartition(count, -2)[-2:]
    indices = np.sort(indices)

    for index in indices:
        bands.append((freq_set[index],freq_set[index+1]))

    return bands

def choosesidechain(selected_stem):
    # it is a flexible function, maybe not be used to select side chain. 
    list = []
    if selected_stem == "vocals":
       result1 = "left_"+"vocals"
       list.append(result1)
       result2 = "right_"+"vocals"
       list.append(result2)
    elif selected_stem == "drums":
       result1 = "left_"+"drums"
       list.append(result1)
       result2 = "right_"+"drums"
       list.append(result2)
    elif selected_stem == "bass":
        result1 = "left_" + "bass"
        list.append(result1)
        result2 = "right_" + "bass"
        list.append(result2)
    else:
        result1 = "left_" + "other"
        list.append(result1)
        result2 = "right_" + "other"
        list.append(result2)

    return list

def choosecompressor(stem_str,config):
    if stem_str == "left_vocals" or stem_str == "right_vocals":
       selected_compressor = Compressor(**config.compressor_vocals)
    elif stem_str == "left_bass" or stem_str == "right_bass":
       selected_compressor = Compressor(**config.compressor_bass)
    elif stem_str == "left_drums" or stem_str == "right_drums":
       selected_compressor = Compressor(**config.compressor_drums)
    else:
       selected_compressor = Compressor(**config.compressor_other)
    return selected_compressor

def remix_signal(stems: dict, audiogram_left,audiogram_right,config) -> np.ndarray:
   
    n_samples = stems[list(stems.keys())[0]].shape[0]
    out_left, out_right = np.zeros(n_samples), np.zeros(n_samples)
    bands = []
            
    for stem_str, stem_signal in stems.items():
        if stem_str in choosesidechain("vocals"):
            bands = selectbands(stem_signal,44100)
            print(bands)
        
    for stem_str, stem_signal in stems.items():
        
        if stem_str.startswith("l") and \
           stem_str not in choosesidechain("vocals") and \
           list(analyze_audiogram(classify_frequency_listeners(audiogram_left)))[0]==True:
              
            multicompressor = MultibandCompressor(bands)
            compressed_signal = multicompressor.process(stem_signal,44100, choosecompressor(stem_str, config))
            stem_signal = compressed_signal
            
                            
        elif stem_str.startswith("r") and \
             stem_str not in choosesidechain("vocals") and \
             list(analyze_audiogram(classify_frequency_listeners(audiogram_right)))[0]==True:
        
            multicompressor = MultibandCompressor(bands)
            compressed_signal = multicompressor.process(stem_signal,44100, choosecompressor(stem_str, config))
            stem_signal = compressed_signal
           
                
        if stem_str.startswith("l"):
            print(stem_str)
            db2 = list(analyze_audiogram(classify_frequency_listeners(audiogram_left)))[1]
            db3 = list(analyze_audiogram(classify_frequency_listeners(audiogram_left)))[2]
            db4 = list(analyze_audiogram(classify_frequency_listeners(audiogram_left)))[3]
            stem_signal = judge_volume(stem_str,stem_signal,0,db2,db3,db4)
            out_left += stem_signal
            
        else:
            print(stem_str)
            db2 = list(analyze_audiogram(classify_frequency_listeners(audiogram_right)))[1]
            db3 = list(analyze_audiogram(classify_frequency_listeners(audiogram_right)))[2]
            db4 = list(analyze_audiogram(classify_frequency_listeners(audiogram_right)))[3]
            stem_signal = judge_volume(stem_str,stem_signal,0,db2,db3,db4)
            out_right += stem_signal
                
    
    return np.stack([out_left, out_right], axis=1)




@hydra.main(config_path="", config_name="config")
def enhance(config: DictConfig) -> None:
    """
    Run the music enhancement.
    The system decomposes the music into vocal, drums, bass, and other stems.
    Then, the NAL-R prescription procedure is applied to each stem.
    Args:
        config (dict): Dictionary of configuration options for enhancing music.

    Returns 8 stems for each song:
        - left channel vocal, drums, bass, and other stems
        - right channel vocal, drums, bass, and other stems
    """

    if config.separator.model not in ["demucs", "openunmix"]:
        raise ValueError(f"Separator model {config.separator.model} not supported.")

    enhanced_folder = Path("enhanced_signals") / "evaluation"
    enhanced_folder.mkdir(parents=True, exist_ok=True)

    if config.separator.model == "demucs":
        separation_model = HDEMUCS_HIGH_MUSDB.get_model()
        model_sample_rate = HDEMUCS_HIGH_MUSDB.sample_rate
        sources_order = separation_model.sources
        normalise = True
    elif config.separator.model == "openunmix":
        separation_model = torch.hub.load("sigsep/open-unmix-pytorch", "umxhq", niter=0)
        model_sample_rate = separation_model.sample_rate
        sources_order = ["vocals", "drums", "bass", "other"]
        normalise = False
    else:
        raise ValueError(f"Separator model {config.separator.model} not supported.")

    device, _ = get_device(config.separator.device)
    separation_model.to(device)

    # Processing Validation Set
    # Load listener audiograms and songs
    with open(config.path.listeners_test_file, encoding="utf-8") as file:
        listener_audiograms = json.load(file)

    with open(config.path.music_test_file, encoding="utf-8") as file:
        song_data = json.load(file)
    songs_details = pd.DataFrame.from_dict(song_data)

    with open(config.path.music_segments_test_file, encoding="utf-8") as file:
        songs_segments = json.load(file)

    song_listener_pairs = make_song_listener_list(
        songs_details["Track Name"], listener_audiograms
    )
    # Select a batch to process
    song_listener_pairs = song_listener_pairs[
        config.evaluate.batch :: config.evaluate.batch_size
    ]

    # Create hearing aid objects
    enhancer = NALR(**config.nalr)
    compressor = Compressor(**config.compressor)

    # Decompose each song into left and right vocal, drums, bass, and other stems
    # and process each stem for the listener
    prev_song_name = None
    num_song_list_pair = len(song_listener_pairs)
    for idx, song_listener in enumerate(song_listener_pairs, 1):
        song_name, listener_name = song_listener
        logger.info(
            f"[{idx:03d}/{num_song_list_pair:03d}] "
            f"Processing {song_name} for {listener_name}..."
        )
        # Get the listener's audiogram
        listener_info = listener_audiograms[listener_name]

        # Find the music split directory
        split_directory = (
            "test"
            if songs_details.loc[
                songs_details["Track Name"] == song_name, "Split"
            ].iloc[0]
            == "test"
            else "train"
        )

        critical_frequencies = np.array(listener_info["audiogram_cfs"])
        audiogram_left = np.array(listener_info["audiogram_levels_l"])
        audiogram_right = np.array(listener_info["audiogram_levels_r"])

        # Baseline Steps
        # 1. Decompose the mixture signal into vocal, drums, bass, and other stems
        #    We validate if 2 consecutive signals are the same to avoid
        #    decomposing the same song multiple times
        if prev_song_name != song_name:
            # Decompose song only once
            prev_song_name = song_name

            sample_rate, mixture_signal = wavfile.read(
                Path(config.path.music_dir)
                / split_directory
                / song_name
                / "mixture.wav"
            )
            mixture_signal = (mixture_signal / 32768.0).astype(np.float32).T
            assert sample_rate == config.sample_rate

            # Decompose mixture signal into stems
            stems = decompose_signal(
                separation_model,
                model_sample_rate,
                mixture_signal,
                sample_rate,
                device,
                sources_order,
                audiogram_left,
                audiogram_right,
                normalise,
            )

        # 2. Apply NAL-R prescription to each stem
        #     Baseline applies NALR prescription to each stem instead of using the
        #     listener's audiograms in the decomposition. This step can be skipped
        #     if the listener's audiograms are used in the decomposition
        processed_stems = process_stems_for_listener(
            stems,
            enhancer,
            compressor,
            audiogram_left,
            audiogram_right,
            critical_frequencies,
            config.apply_compressor,
        )

        # 3. Save processed stems
        for stem_str, stem_signal in processed_stems.items():
            filename = (
                enhanced_folder
                / f"{listener_name}"
                / f"{song_name}"
                / f"{listener_name}_{song_name}_{stem_str}.flac"
            )
            filename.parent.mkdir(parents=True, exist_ok=True)
            start = songs_segments[song_name]["objective_evaluation"]["start"]
            end = songs_segments[song_name]["objective_evaluation"]["end"]
            save_flac_signal(
                signal=stem_signal[
                    int(start * config.sample_rate) : int(end * config.sample_rate)
                ],
                filename=filename,
                signal_sample_rate=config.sample_rate,
                output_sample_rate=config.stem_sample_rate,
                do_scale_signal=True,
            )

        # 3. Remix Signal
        enhanced = remix_signal(processed_stems,audiogram_left,audiogram_right,config)

        # 5. Save enhanced (remixed) signal
        filename = (
            enhanced_folder
            / f"{listener_info['name']}"
            / f"{song_name}"
            / f"{listener_info['name']}_{song_name}_remix.flac"
        )
        start = songs_segments[song_name]["subjective_evaluation"]["start"]
        end = songs_segments[song_name]["subjective_evaluation"]["end"]
        save_flac_signal(
            signal=enhanced[
                int(start * config.sample_rate) : int(end * config.sample_rate)
            ],
            filename=filename,
            signal_sample_rate=config.sample_rate,
            output_sample_rate=config.remix_sample_rate,
            do_clip_signal=True,
            do_soft_clip=config.soft_clip,
        )

    pack_submission(
        team_id=config.team_id,
        root_dir=enhanced_folder.parent,
        base_dir=enhanced_folder.name,
    )

    logger.info("Evaluation complete.!!")
    logger.info(
        f"Please, submit the file submission_{config.team_id}.zip to the challenge "
        "using the link provided. Thank you.!!"
    )


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    enhance()

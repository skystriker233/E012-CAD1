""" Run the baseline enhancement. """
from __future__ import annotations

# pylint: disable=import-error
import json
import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from scipy.io import wavfile
from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB
from torchaudio.transforms import Fade

from clarity.enhancer.compressor import Compressor
from clarity.enhancer.nalr import NALR
from clarity.utils.flac_encoder import FlacEncoder
from clarity.utils.signal_processing import (
    clip_signal,
    denormalize_signals,
    normalize_signal,
    to_16bit,
)
from recipes.cad1.task1.baseline.evaluate import make_song_listener_list, resample
from multibandCompressor import MultibandCompressor

# pylint: disable=too-many-locals

logger = logging.getLogger(__name__)


def separate_sources(
    model: torch.nn.Module,
    mix: torch.Tensor | np.ndarray,
    sample_rate: int,
    segment: float = 10.0,
    overlap: float = 0.1,
    device: torch.device | str = None,
):
    """
    Apply model to a given mixture.
    Use fade, and add segments together in order to add model segment by segment.

    Args:
        model (torch.nn.Module): model to use for separation
        mix (torch.Tensor): mixture to separate, shape (batch, channels, time)
        sample_rate (int): sampling rate of the mixture
        segment (float): segment length in seconds
        overlap (float): overlap between segments, between 0 and 1
        device (torch.device, str, or None): if provided, device on which to
            execute the computation, otherwise `mix.device` is assumed.
            When `device` is different from `mix.device`, only local computations will
            be on `device`, while the entire tracks will be stored on `mix.device`.

    Returns:
        torch.Tensor: estimated sources

    Based on https://pytorch.org/audio/main/tutorials/hybrid_demucs_tutorial.html
    """
    device = mix.device if device is None else torch.device(device)
    mix = torch.as_tensor(mix, device=device)

    if mix.ndim == 1:
        # one track and mono audio
        mix = mix.unsqueeze(0)
    elif mix.ndim == 2:
        # one track and stereo audio
        mix = mix.unsqueeze(0)

    batch, channels, length = mix.shape

    chunk_len = int(sample_rate * segment * (1 + overlap))
    start = 0
    end = chunk_len
    overlap_frames = overlap * sample_rate
    fade = Fade(fade_in_len=0, fade_out_len=int(overlap_frames), fade_shape="linear")

    final = torch.zeros(batch, 4, channels, length, device=device)

    while start < length - overlap_frames:
        chunk = mix[:, :, start:end]
        with torch.no_grad():
            out = model.forward(chunk)
        out = fade(out)
        final[:, :, :, start:end] += out
        if start == 0:
            fade.fade_in_len = int(overlap_frames)
            start += int(chunk_len - overlap_frames)
        else:
            start += chunk_len
        end += chunk_len
        if end >= length:
            fade.fade_out_len = 0

    return final.cpu().detach().numpy()


def get_device(device: str) -> tuple:
    """Get the Torch device.

    Args:
        device (str): device type, e.g. "cpu", "gpu0", "gpu1", etc.

    Returns:
        torch.device: torch.device() appropiate to the hardware available.
        str: device type selected, e.g. "cpu", "cuda".
    """
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda"), "cuda"
        return torch.device("cpu"), "cpu"

    if device.startswith("gpu"):
        device_index = int(device.replace("gpu", ""))
        if device_index > torch.cuda.device_count():
            raise ValueError(f"GPU device index {device_index} is not available.")
        return torch.device(f"cuda:{device_index}"), "cuda"

    if device == "cpu":
        return torch.device("cpu"), "cpu"

    raise ValueError(f"Unsupported device type: {device}")


def map_to_dict(sources: np.ndarray, sources_list: list[str]) -> dict:
    """Map sources to a dictionary separating audio into left and right channels.

    Args:
       sources (np.ndarray): Signal to be mapped to dictionary.
       sources_list (list): List of strings used to index dictionary.

    Returns:
        Dictionary: A dictionary of separated source audio split into channels.
    """
    audios = dict(zip(sources_list, sources))

    signal_stems = {}
    for source in sources_list:
        audio = audios[source]
        signal_stems[f"left_{source}"] = audio[0]
        signal_stems[f"right_{source}"] = audio[1]

    return signal_stems


# pylint: disable=unused-argument
def decompose_signal(
    model: torch.nn.Module,
    model_sample_rate: int,
    signal: np.ndarray,
    signal_sample_rate: int,
    device: torch.device,
    sources_list: list[str],
    left_audiogram: np.ndarray,
    right_audiogram: np.ndarray,
    normalise: bool = True,
) -> dict[str, np.ndarray]:
    """
    Decompose signal into 8 stems.

    The left and right audiograms are ignored by the baseline system as it
    is performing personalised decomposition.
    Instead, it performs a standard music decomposition using the
    HDEMUCS model trained on the MUSDB18 dataset.

    Args:
        model (torch.nn.Module): Torch model.
        model_sample_rate (int): Sample rate of the model.
        signal (np.ndarray): Signal to be decomposed.
        signal_sample_rate (int): Sample frequency.
        device (torch.device): Torch device to use for processing.
        sources_list (list): List of strings used to index dictionary.
        left_audiogram (np.ndarray): Left ear audiogram.
        right_audiogram (np.ndarray): Right ear audiogram.
        normalise (bool): Whether to normalise the signal.

     Returns:
         Dictionary: Indexed by sources with the associated model as values.
    """

    # Resample mixture signal to model sample rate
    if signal_sample_rate != model_sample_rate:
        signal = resample(signal, signal_sample_rate, model_sample_rate)

    if normalise:
        signal, ref = normalize_signal(signal)

    sources = separate_sources(
        model, torch.from_numpy(signal), signal_sample_rate, device=device
    )
    # only one element in the batch
    sources = sources[0]

    if normalise:
        sources = denormalize_signals(sources, ref)

    signal_stems = map_to_dict(sources, sources_list)
    return signal_stems


def apply_baseline_ha(
    enhancer: NALR,
    compressor: Compressor,
    signal: np.ndarray,
    listener_audiogram: np.ndarray,
    cfs: np.ndarray,
    apply_compressor: bool = False,
) -> np.ndarray:
    """
    Apply NAL-R prescription hearing aid to a signal.

    Args:
        enhancer: A NALR object that enhances the signal.
        compressor: A Compressor object that compresses the signal.
        signal: An ndarray representing the audio signal.
        listener_audiogram: An ndarray representing the listener's audiogram.
        cfs: An ndarray of center frequencies.
        apply_compressor: A boolean indicating whether to include the compressor.

    Returns:
        An ndarray representing the processed signal.
    """
    nalr_fir, _ = enhancer.build(listener_audiogram, cfs)
    proc_signal = enhancer.apply(nalr_fir, signal)
    if apply_compressor:
        proc_signal, _, _ = compressor.process(proc_signal)
    return proc_signal


def process_stems_for_listener(
    stems: dict,
    enhancer: NALR,
    compressor: Compressor,
    audiogram_left: np.ndarray,
    audiogram_right: np.ndarray,
    cfs: np.ndarray,
    apply_compressor: bool = False,
) -> dict:
    """Process the stems from sources.

    Args:
        stems (dict) : Dictionary of stems
        enhancer (NALR) : NAL-R prescription hearing aid
        compressor (Compressor) : Compressor
        audiogram_left (np.ndarray) : Left channel audiogram
        audiogram_right (np.ndarray) : Right channel audiogram
        cfs (np.ndarray) : Center frequencies
        apply_compressor (bool) : Whether to apply the compressor

    Returns:
        processed_sources (dict) : Dictionary of processed stems
    """

    processed_stems = {}

    for stem_str in stems:
        stem_signal = stems[stem_str]

        # Determine the audiogram to use
        audiogram = audiogram_left if stem_str.startswith("l") else audiogram_right

        # Apply NALR prescription to stem_signal
        proc_signal = apply_baseline_ha(
            enhancer, compressor, stem_signal, audiogram, cfs, apply_compressor
        )
        processed_stems[stem_str] = proc_signal
    return processed_stems

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
    # 0(None): <20 ,  1(mild): 20-34,  2(moderate): 35-49, 3(moderately severe):50-64, 4(severe): >65

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
    
    

def a1_coefficient(break_frequency, sampling_rate):
    tan = np.tan(np.pi * break_frequency / sampling_rate)
    return (tan - 1) / (tan + 1)
    
def allpass_filter(input_signal, break_frequency, sampling_rate):
   
    allpass_output = np.zeros_like(input_signal)
    dn_1 = 0

    for n in range(input_signal.shape[0]):
        
        a1 = a1_coefficient(break_frequency[n], sampling_rate)
        allpass_output[n] = a1 * input_signal[n] + dn_1

        dn_1 = input_signal[n] - a1 * allpass_output[n]
    return allpass_output

def allpass_based_filter(input_signal, cutoff_frequency, \
    sampling_rate, highpass=False, amplitude=1.0):
    
    allpass_output = allpass_filter(input_signal, \
        cutoff_frequency, sampling_rate)

    # For highpass filter
    if highpass:
        allpass_output *= -1

    filter_output = input_signal + allpass_output
    filter_output *= 0.5
    filter_output *= amplitude

    return filter_output 
    
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
    """
    Function to remix signal. It takes the eight stems
    and combines them into a stereo signal.

    Args:
        stems (dict) : Dictionary of stems

    Returns:
        (np.ndarray) : Remixed signal

    """
    n_samples = stems[list(stems.keys())[0]].shape[0]
    out_left, out_right = np.zeros(n_samples), np.zeros(n_samples)
    bands = [(500,4000)]
    print(list(analyze_audiogram(classify_frequency_listeners(audiogram_left)))[0])
    print(list(analyze_audiogram(classify_frequency_listeners(audiogram_left)))[1])
    print(list(analyze_audiogram(classify_frequency_listeners(audiogram_left)))[2])
    print(list(analyze_audiogram(classify_frequency_listeners(audiogram_left)))[3])
    print(list(analyze_audiogram(classify_frequency_listeners(audiogram_right)))[0])
    print(list(analyze_audiogram(classify_frequency_listeners(audiogram_right)))[1])
    print(list(analyze_audiogram(classify_frequency_listeners(audiogram_right)))[2])
    print(list(analyze_audiogram(classify_frequency_listeners(audiogram_right)))[3])
       
        
        
    #for stem_str, stem_signal in stems.items():
        #if stem_str in choosesidechain("vocals"):
            #bands = selectbands(stem_signal,44100)
            #print(bands)
    
    
    for stem_str, stem_signal in stems.items():
        
        if stem_str.startswith("l") and stem_str in choosesidechain("other") and list(analyze_audiogram(classify_frequency_listeners(audiogram_left)))[0]==True:
              
            multicompressor = MultibandCompressor(bands)
            compressed_signal = multicompressor.process(stem_signal,44100, choosecompressor(stem_str, config))
            stem_signal = compressed_signal
            
                            
        elif stem_str.startswith("r") and stem_str in choosesidechain("other") and list(analyze_audiogram(classify_frequency_listeners(audiogram_right)))[0]==True:
        
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

def save_flac_signal(
    signal: np.ndarray,
    filename: Path,
    signal_sample_rate,
    output_sample_rate,
    do_clip_signal: bool = False,
    do_soft_clip: bool = False,
    do_scale_signal: bool = False,
) -> None:
    """
    Function to save output signals.

    - The output signal will be resample to ``output_sample_rate``
    - The output signal will be clipped to [-1, 1] if ``do_clip_signal`` is True
        and use soft clipped if ``do_soft_clip`` is True. Note that if
        ``do_clip_signal`` is False, ``do_soft_clip`` will be ignored.
        Note that if ``do_clip_signal`` is True, ``do_scale_signal`` will be ignored.
    - The output signal will be scaled to [-1, 1] if ``do_scale_signal`` is True.
        If signal is scale, the scale factor will be saved in a TXT file.
        Note that if ``do_clip_signal`` is True, ``do_scale_signal`` will be ignored.
    - The output signal will be saved as a FLAC file.

    Args:
        signal (np.ndarray) : Signal to save
        filename (Path) : Path to save signal
        signal_sample_rate (int) : Sample rate of the input signal
        output_sample_rate (int) : Sample rate of the output signal
        do_clip_signal (bool) : Whether to clip signal
        do_soft_clip (bool) : Whether to apply soft clipping
        do_scale_signal (bool) : Whether to scale signal
    """
    # Resample signal to expected output sample rate
    if signal_sample_rate != output_sample_rate:
        signal = resample(signal, signal_sample_rate, output_sample_rate)

    if do_scale_signal:
        # Scale stem signal
        max_value = np.max(np.abs(signal))
        signal = signal / max_value

        # Save scale factor
        with open(filename.with_suffix(".txt"), "w", encoding="utf-8") as file:
            file.write(f"{max_value}")

    elif do_clip_signal:
        # Clip the signal
        signal, n_clipped = clip_signal(signal, do_soft_clip)
        if n_clipped > 0:
            logger.warning(f"Writing {filename}: {n_clipped} samples clipped")

    # Convert signal to 16-bit integer
    signal = to_16bit(signal)

    # Create flac encoder object to compress and save the signal
    FlacEncoder().encode(signal, output_sample_rate, filename)



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

    enhanced_folder = Path("enhanced_signals")
    enhanced_folder.mkdir(parents=True, exist_ok=True)

    # Training stage
    #
    # The baseline is using an off-the-shelf model trained on the MUSDB18 dataset
    # Training listeners and song are not necessary in this case.
    #
    # Training songs and audiograms can be read like this:
    #
    #  with open(config.path.listeners_train_file, "r", encoding="utf-8") as file:
    #        listener_train_audiograms = json.load(file)
    #
    #  with open(config.path.music_train_file, "r", encoding="utf-8") as file:
    #        song_data = json.load(file)
    #  songs_train = pd.DataFrame.from_dict(song_data)
    #
    # train_song_listener_pairs = make_song_listener_list(
    #     songs_train['Track Name'], listener_train_audiograms
    # )

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
    with open(config.path.listeners_valid_file, encoding="utf-8") as file:
        listener_valid_audiograms = json.load(file)

    with open(config.path.music_valid_file, encoding="utf-8") as file:
        song_data = json.load(file)
    songs_valid = pd.DataFrame.from_dict(song_data)

    valid_song_listener_pairs = make_song_listener_list(
        songs_valid["Track Name"], listener_valid_audiograms
    )
    # Select a batch to process
    valid_song_listener_pairs = valid_song_listener_pairs[
        config.evaluate.batch :: config.evaluate.batch_size
    ]

    # Create hearing aid objects
    enhancer = NALR(**config.nalr)
    compressor = Compressor(**config.compressor)

    # Decompose each song into left and right vocal, drums, bass, and other stems
    # and process each stem for the listener
    prev_song_name = None
    num_song_list_pair = len(valid_song_listener_pairs)
    for idx, song_listener in enumerate(valid_song_listener_pairs, 1):
        song_name, listener_name = song_listener
        logger.info(
            f"[{idx:03d}/{num_song_list_pair:03d}] "
            f"Processing {song_name} for {listener_name}..."
        )
        # Get the listener's audiogram
        listener_info = listener_valid_audiograms[listener_name]

        # Find the music split directory
        split_directory = (
            "test"
            if songs_valid.loc[songs_valid["Track Name"] == song_name, "Split"].iloc[0]
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
            save_flac_signal(
                signal=stem_signal,
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
        save_flac_signal(
            signal=enhanced,
            filename=filename,
            signal_sample_rate=config.sample_rate,
            output_sample_rate=config.remix_sample_rate,
            do_clip_signal=True,
            do_soft_clip=config.soft_clip,
        )


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    enhance()

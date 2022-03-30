import argparse
import pathlib
from tqdm import tqdm

import math
import os

import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0

def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate

def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        #print("fmax not in mel_basis")
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    #print(y.size())
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)
    #rint(spec.size())

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)
    #print(spec.size())

    return spec

def create_mel_file(wavfile, melfile, output_melfile, output_melloss_file, output_posfile, flags):
    hop_size         = flags.hop_size
    segment_size     = flags.segment_size
    segment_hop_size = flags.segment_hop_size
    segment_frames   = flags.segment_frames
    frame_hop_size   = flags.frame_hop_size
    n_fft            = flags.fft_size
    num_mels         = flags.num_mels
    win_size         = flags.fft_size
    fmin             = flags.fmin
    fmax             = flags.fmax
    fmax_loss        = None

    # load wav
    print("wavfile path:", wavfile)
    audio, sampling_rate = load_wav(wavfile)
    print("audio shape:", audio.shape)
    print("sampling rate:", sampling_rate)

    audio = audio / MAX_WAV_VALUE
    audio = normalize(audio) * 0.95
    
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)

    audio_size = audio.size(1)
    segment_cnt = (audio_size - segment_size) // segment_hop_size - 1

    input_mel = np.load(melfile)
    print("generated mel shape:", input_mel.shape)
    print("audio samples, num of segments:", audio_size, segment_cnt)

    # setup buffer
    mel_shape = (segment_cnt, num_mels, segment_frames)
    pos_shape = (segment_cnt, 2)
    mel_buffer = np.zeros(shape=mel_shape, dtype=np.float32)
    mel_loss_buffer = np.zeros(shape=mel_shape, dtype=np.float32)
    pos_buffer = np.zeros(shape=pos_shape, dtype=np.int32)

    for segment_idx in tqdm(range(segment_cnt)):
        start_pt = segment_idx * segment_hop_size
        end_pt   = start_pt + segment_size
        segment_data = audio[:,start_pt:end_pt]

        start_mel_pt = segment_idx * frame_hop_size
        end_mel_pt   = start_mel_pt + segment_frames

        mel = torch.FloatTensor(input_mel[:,:,start_mel_pt:end_mel_pt])
        #print(mel.size())
        #mel = mel_spectrogram(segment_data, n_fft, num_mels,
        #    sampling_rate, hop_size, win_size, fmin, fmax,
        #    center=False)
        #print(mel.size())
        mel_loss = mel_spectrogram(segment_data, n_fft, num_mels,
            sampling_rate, hop_size, win_size, fmin, fmax_loss,
            center=False)

        #print(segment_data.size())
        #print(mel.size())
        #print(mel_loss.size())

        np_mel = mel.numpy()
        np_mel_loss = mel_loss.numpy()
        np_pos = np.array([start_pt, end_pt], dtype=np.int32)

        mel_buffer[segment_idx] = np_mel[0]
        mel_loss_buffer[segment_idx] = np_mel_loss[0]
        pos_buffer[segment_idx] = np_pos

        #print(np_mel)
        #print(mel_buffer[segment_idx])
        #exit()

        # the shapes of mel, mel_loss are [1, 80, 32]
    
    # save numpy buffer to files
    np.savez(output_melfile, mel_buffer)
    np.savez(output_melloss_file, mel_loss_buffer)
    np.savez(output_posfile, pos_buffer)
    pass

def main(args):
    wavfile = pathlib.Path(args.wavfile)
    melfile = pathlib.Path(args.melfile)
    outputdir = pathlib.Path(args.outputdir)
    wav_basename = wavfile.stem

    output_melfile      = outputdir / ("%s_mel.npz" % wav_basename)
    output_melloss_file = outputdir / ("%s_melloss.npz" % wav_basename)
    output_posfile      = outputdir / ("%s_pos.npz" % wav_basename)

    create_mel_file(wavfile, melfile, output_melfile, output_melloss_file, output_posfile, args)
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wavfile")
    parser.add_argument("melfile")
    parser.add_argument("outputdir")
    parser.add_argument("--segment_size", default=8192)
    parser.add_argument("--segment_hop_size", default=4096)
    parser.add_argument("--segment_frames", default=32)
    parser.add_argument("--frame_hop_size", default=16)
    parser.add_argument("--fft_size", default=1024)
    parser.add_argument("--hop_size", default=256)
    parser.add_argument("--num_mels", default=80)
    parser.add_argument("--fmin", default=0)
    parser.add_argument("--fmax", default=8000)

    args = parser.parse_args()
    main(args)
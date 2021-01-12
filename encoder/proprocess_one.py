from encoder.params_data import *
from encoder import audio
from pathlib import Path
import numpy as np

def preprocess(in_fpath, out_fpath, parent_path):

    source_text = parent_path / "_sources.txt"
    sources_file = source_text.open("w")

    # Load and preprocess the waveform
    wav = audio.preprocess_wav(in_fpath)
    if len(wav) == 0:
        print("empty audio file")
    
    # Create the mel spectrogram, discard those that are too short
    frames = audio.wav_to_mel_spectrogram(wav)
    if len(frames) < partials_n_frames:
        print("{} < {}, number of frames is less than partials_n_frames".format(len(frames), partials_n_frames))

    np.save(out_fpath, frames)
    sources_file.write("%s,%s\n" % (out_fpath.name + '.npy', in_fpath.name))
    
    sources_file.close()

    return frames
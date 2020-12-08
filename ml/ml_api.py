from flask_restx import Resource
from flask import request, send_file
from bson.objectid import ObjectId
import inflect
import math
import translators as ts

# from app import db_connection
from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from utils.modelutils import check_model_paths
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import argparse
import torch
import sys


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("-e", "--enc_model_fpath", type=Path, 
                    default="encoder/saved_models/pretrained_ft_backup/pretrained_bak_1565500.pt",
                    help="Path to a saved encoder")
parser.add_argument("-s", "--syn_model_dir", type=Path, 
                   default="synthesizer/saved_models/315k_0005/",
                   help="Directory containing the synthesizer model")
parser.add_argument("-v", "--voc_model_fpath", type=Path, 
                   default="vocoder/saved_models/pretrained/pretrained.pt",
                   help="Path to a saved vocoder")
parser.add_argument("--low_mem", action="store_true", help=\
    "If True, the memory used by the synthesizer will be freed after each use. Adds large "
    "overhead but allows to save some GPU memory for lower-end GPUs.")
parser.add_argument("--no_sound", action="store_true", help=\
    "If True, audio won't be played.")
parser.add_argument("--seed", type=int, default=None, help=\
    "Optional random number seed value to make toolbox deterministic.")
parser.add_argument("--no_mp3_support", action="store_true", help=\
    "If True, disallows loading mp3 files to prevent audioread errors when ffmpeg is not installed.")
args = parser.parse_args()

print_args(args, parser)
if not args.no_sound:
    import sounddevice as sd

if not args.no_mp3_support:
    try:
        librosa.load("samples/1320_00000.mp3")
    except NoBackendError:
        print("Librosa will be unable to open mp3 files if additional software is not installed.\n"
              "Please install ffmpeg or add the '--no_mp3_support' option to proceed without support for mp3 files.")
        exit(-1)

print("Running a test of your configuration...\n")

if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(device_id)
    ## Print some environment information (for debugging purposes)
    print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
        "%.1fGb total memory.\n" % 
        (torch.cuda.device_count(),
        device_id,
        gpu_properties.name,
        gpu_properties.major,
        gpu_properties.minor,
        gpu_properties.total_memory / 1e9))
else:
    print("Using CPU for inference.\n")

## Remind the user to download pretrained models if needed
check_model_paths(encoder_path=args.enc_model_fpath, synthesizer_path=args.syn_model_dir,
                  vocoder_path=args.voc_model_fpath)

## Load the models one by one.
print("Preparing the encoder, the synthesizer and the vocoder...")
encoder.load_model(args.enc_model_fpath)
synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=args.low_mem, seed=args.seed)
vocoder.load_model(args.voc_model_fpath)


# this api instance is make random number of recipe for front page
class ML_Voice_Generate(Resource):

    def post(self):
        #print(request.__dict__)
        post_data = request.get_json()
        input_text = post_data.get('text', 'None')
        print(input_text)
        text = ts.alibaba(input_text)

        print(text)
        # use the text to and model to get the audio
        # TODO

        in_fpath = '/home/ubuntu/Real-Time-Voice-Cloning/samples/my_sample_01.mp3'

        preprocessed_wav = encoder.preprocess_wav(in_fpath)
        # - If the wav is already loaded:
        original_wav, sampling_rate = librosa.load(str(in_fpath))
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        print("Loaded file succesfully")

        # Then we derive the embedding. There are many functions and parameters that the 
        # speaker encoder interfaces. These are mostly for in-depth research. You will typically
        # only use this function (with its default parameters):
        embed = encoder.embed_utterance(preprocessed_wav)
        print("Created the embedding")
        
        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        texts = [text]
        embeds = [embed]
        # If you know what the attention layer alignments are, you can retrieve them here by
        # passing return_alignments=True
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        print("Created the mel spectrogram")


        ## Generating the waveform
        print("Synthesizing the waveform:")

        # If seed is specified, reset torch seed and reload vocoder
        if args.seed is not None:
            torch.manual_seed(args.seed)
            vocoder.load_model(args.voc_model_fpath)

        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        generated_wav = vocoder.infer_waveform(spec)

        ## Post-generation
        # There's a bug with sounddevice that makes the audio cut one second earlier, so we
        # pad it.
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

        # Trim excess silences to compensate for gaps in spectrograms (issue #53)
        generated_wav = encoder.preprocess_wav(generated_wav)
        
        sf.write("test_file.wav", generated_wav.astype(np.float32), synthesizer.sample_rate)

        return send_file("/home/ubuntu/Real-Time-Voice-Cloning/test_file.wav"), 200



# this api instance is make random number of recipe for front page
class ML_Fine_Tune(Resource):

    def post(self):
        post_data = request.get_json()
        # todo here it should be a file
        input_text = post_data.get('voice', None)

        # use the audio to do the fine tuning
        # TODO

        return {'result':'ML_Fine_Tune'}, 200

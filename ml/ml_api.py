from flask_restx import Resource
from flask import request, send_from_directory

from bson.objectid import ObjectId
import inflect
import math
from googletrans import Translator
import time

import asyncio
import threading
import uuid

# from app import db_connection
from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from utils.modelutils import check_model_paths
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from encoder.proprocess_one import preprocess
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import argparse
import torch
import sys

# fine tuning
from shutil import copyfile
from encoder_train import encoder_train as fine_tune

# create instance
translator = Translator()

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

# ================== load models ==================
start_time = time.time()

## Load the models one by one.
print("Preparing the encoder, the synthesizer and the vocoder...")
encoder.load_model(args.enc_model_fpath)
synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=False, seed=args.seed)
vocoder.load_model(args.voc_model_fpath)
print("--- load models: %s seconds ---" % (time.time() - start_time))


class Get_file(Resource):
    def post(self, filename):
        print(filename)

        return send_from_directory('/home/ubuntu/Real-Time-Voice-Cloning/samples', \
            filename=filename, as_attachment=True)


class save_file(Resource):
    def post(self):
        print(request.__dict__)
        file = request.files.get('audio')

        print(file)

        file.save("/home/ubuntu/Real-Time-Voice-Cloning/samples/" + file.filename);


        return 'succussful', 200

@asyncio.coroutine
async def generate_wav(text, filename):
    user_id = "russell"
    embed_path = "user_data/embeds/{}.npy".format(user_id)
    embed_path = Path(embed_path)

    if embed_path.is_file():
        embed = np.load(embed_path)
        print("load embedding in {}".format(embed_path))
    else:
        raise("user embedding not found")

    # ================== synthesizer ==================
    start_time = time.time()
    
    # The synthesizer works in batch, so you need to put your data in a list or numpy array
    texts = [text]
    embeds = [embed]
    # If you know what the attention layer alignments are, you can retrieve them here by
    # passing return_alignments=True
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    spec = specs[0]
    print("Created the mel spectrogram")

    print("--- synthesizer: %s seconds ---" % (time.time() - start_time))


    # ================== vocoder ==================
    start_time = time.time()

    # If seed is specified, reset torch seed and reload vocoder
    if args.seed is not None:
        torch.manual_seed(args.seed)
        vocoder.load_model(args.voc_model_fpath)

    # Synthesizing the waveform is fairly straightforward. Remember that the longer the
    # spectrogram, the more time-efficient the vocoder.
    generated_wav = vocoder.infer_waveform(spec)
    print("")
    print("--- vocoder: %s seconds ---" % (time.time() - start_time))


    # ================== post generation ==================
    start_time = time.time()

    # There's a bug with sounddevice that makes the audio cut one second earlier, so we
    # pad it.
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

    # Trim excess silences to compensate for gaps in spectrograms (issue #53)
    generated_wav = encoder.preprocess_wav(generated_wav)
    print("--- post generation: %s seconds ---" % (time.time() - start_time))
    
    sf.write("./user_data/generated_voice/%s/"%(user_id) + "%s.wav"%filename, \
            generated_wav.astype(np.float32), synthesizer.sample_rate)

def loop_in_thread(loop, text, filename):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(generate_wav(text, filename))

class Translator_Api(Resource):
    def post(self):
        print("====== Translator ======")
        post_data = request.get_json()
        input_text = post_data.get('text', 'None')
        print(input_text)
        text = translator.translate(input_text).text
        print(text)

        # generate a hash tag for voice name
        hash_uuid = str(uuid.uuid4())

        loop = asyncio.new_event_loop()
        t = threading.Thread(target=loop_in_thread, args=(loop, text, hash_uuid,))
        t.start()

        # wait a little bit
        [x for x in range(100000)]

        return {"text":text, "filename": hash_uuid+".mp3"}, 200

# this api instance is make random number of recipe for front page
class ML_Voice_Generate(Resource):

    def post(self):
        print("******************* generating new translations *******************")
        #print(request.__dict__)
        # ================== translation ==================
        start_time = time.time()

        post_data = request.get_json()
        input_text = post_data.get('text', 'None')
        print(input_text)
        text = translator.translate(input_text).text
        print(text)
        print("--- translation: %s seconds ---" % (time.time() - start_time))

        # ================== load embedding ==================
        user_id = "russell"
        embed_path = "user_data/embeds/{}.npy".format(user_id)
        embed_path = Path(embed_path)

        if embed_path.is_file():
            embed = np.load(embed_path)
            print("load embedding in {}".format(embed_path))
        else:
            raise("user embedding not found")

        # ================== synthesizer ==================
        start_time = time.time()
        
        # The synthesizer works in batch, so you need to put your data in a list or numpy array
        texts = [text]
        embeds = [embed]
        # If you know what the attention layer alignments are, you can retrieve them here by
        # passing return_alignments=True
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        print("Created the mel spectrogram")

        print("--- synthesizer: %s seconds ---" % (time.time() - start_time))


        # ================== vocoder ==================
        start_time = time.time()

        # If seed is specified, reset torch seed and reload vocoder
        if args.seed is not None:
            torch.manual_seed(args.seed)
            vocoder.load_model(args.voc_model_fpath)

        # Synthesizing the waveform is fairly straightforward. Remember that the longer the
        # spectrogram, the more time-efficient the vocoder.
        generated_wav = vocoder.infer_waveform(spec)
        print("")
        print("--- vocoder: %s seconds ---" % (time.time() - start_time))


        # ================== post generation ==================
        start_time = time.time()

        # There's a bug with sounddevice that makes the audio cut one second earlier, so we
        # pad it.
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

        # Trim excess silences to compensate for gaps in spectrograms (issue #53)
        generated_wav = encoder.preprocess_wav(generated_wav)
        print("--- post generation: %s seconds ---" % (time.time() - start_time))
        
        sf.write("test_file.wav", generated_wav.astype(np.float32), synthesizer.sample_rate)

        return text, 200


# # this api instance is make random number of recipe for front page
# class ML_Voice_Generate(Resource):

#     def post(self):
#         #print(request.__dict__)
#         post_data = request.get_json()
#         input_text = post_data.get('text', 'None')
#         print(input_text)
#         text = translator.translate(input_text).text

#         print(text)
#         # use the text to and model to get the audio
#         # TODO
#         start_time = time.time()
#         in_fpath = '/home/ubuntu/Real-Time-Voice-Cloning/samples/my_sample_01.mp3'

#         preprocessed_wav = encoder.preprocess_wav(in_fpath)
#         # - If the wav is already loaded:
#         original_wav, sampling_rate = librosa.load(str(in_fpath))
#         preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
#         print("Loaded file succesfully")
        
#         print("--- %s seconds ---" % (time.time() - start_time))
#         start_time = time.time()

#         # Then we derive the embedding. There are many functions and parameters that the 
#         # speaker encoder interfaces. These are mostly for in-depth research. You will typically
#         # only use this function (with its default parameters):
#         embed = encoder.embed_utterance(preprocessed_wav)
#         print("Created the embedding")
        
#         # The synthesizer works in batch, so you need to put your data in a list or numpy array
#         texts = [text]
#         embeds = [embed]
#         # If you know what the attention layer alignments are, you can retrieve them here by
#         # passing return_alignments=True
#         specs = synthesizer.synthesize_spectrograms(texts, embeds)
#         spec = specs[0]
#         print("Created the mel spectrogram")

#         print("--- %s seconds ---" % (time.time() - start_time))
#         start_time = time.time()
        
#         ## Generating the waveform
#         print("Synthesizing the waveform:")

#         # If seed is specified, reset torch seed and reload vocoder
#         if args.seed is not None:
#             torch.manual_seed(args.seed)
#             vocoder.load_model(args.voc_model_fpath)

#         # Synthesizing the waveform is fairly straightforward. Remember that the longer the
#         # spectrogram, the more time-efficient the vocoder.
#         generated_wav = vocoder.infer_waveform(spec)

#         ## Post-generation
#         # There's a bug with sounddevice that makes the audio cut one second earlier, so we
#         # pad it.
#         generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

#         # Trim excess silences to compensate for gaps in spectrograms (issue #53)
#         generated_wav = encoder.preprocess_wav(generated_wav)
        
#         print("--- %s seconds ---" % (time.time() - start_time))
#         start_time = time.time()
        
#         sf.write("test_file.wav", generated_wav.astype(np.float32), synthesizer.sample_rate)

#         return "success", 200



# this api instance is make random number of recipe for front page
class ML_Fine_Tune(Resource):

    def post(self):
        print("******************* fine tuning *******************")
        print(request.__dict__)
        file = request.files.get('file')

        print(file)
        #print(file.__dict__)
        print("------")
        # return 'aaaaa', 200

        post_data = request.get_json()
        # todo here it should be a file
        input_text = post_data.get('voice', None)

        # use the audio to do the fine tuning
        # TODO: Fetch user id
        user_id = 'russell' # will be user_id
        user_folder = Path('user_data/recordings/{}'.format(user_id))
        user_folder.mkdir(exist_ok=True, parents=True)

        audio_file = user_folder / '{}.mp3'.format(user_id) # recording saved as user_id.mp3
        saved_np_file = user_folder / '{}'.format(user_id)
        preprocess(audio_file, saved_np_file, user_folder)

        # TODO: Enforce recording to be used in training
        # copyfile(audio_file, '/home/ubuntu/VC_dataset/SV2TTS/encoder/') # replace with new recording

        # fine tuning
        ckpt = fine_tune(user_id, user_folder)
        ckpt_name = Path(ckpt).name
        copyfile(ckpt, 'user_data/models/{}'.format(ckpt_name))

        # reload new encoder model
        encoder.load_model(ckpt)

        # ================== load and preprocess audio files ==================
        start_time = time.time()
        in_fpath = audio_file

        preprocessed_wav = encoder.preprocess_wav(in_fpath)
        # - If the wav is already loaded:
        original_wav, sampling_rate = librosa.load(str(in_fpath))
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        print("Loaded file succesfully")

        print("--- load and preprocess: %s seconds ---" % (time.time() - start_time))

        # ================== generate embedding ==================
        start_time = time.time()
        # Then we derive the embedding. There are many functions and parameters that the 
        # speaker encoder interfaces. These are mostly for in-depth research. You will typically
        # only use this function (with its default parameters):
        embed = encoder.embed_utterance(preprocessed_wav)

        embed_path = "user_data/embeds/{}.npy".format(user_id)
        embed_path = Path(embed_path)
        np.save(str(embed_path), embed)
        print("Created the embedding, and saved in {}".format(embed_path))
        print("--- generate embedding: %s seconds ---" % (time.time() - start_time))


        return {'result':'ML_Fine_Tune'}, 200

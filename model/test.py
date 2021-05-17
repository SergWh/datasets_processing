import librosa
import numpy as np
from pydub import AudioSegment

from cons_notes_data.extract import extract_features_loaded
from model import normalize, to_librosa

name = '/home/moby/PycharmProjects/data/gpt/wav_split/1/bending_up_down_half/bending_up_down_half_1_1.wav'
name_idmt = '/home/moby/PycharmProjects/data/IDMT-SMT-32/dataset2/audio/AR_A_fret_0-20.wav'

audio = AudioSegment.from_file(name, format="wav")[:3000]
y, sr = librosa.load(name, sr=44100)
normalized = normalize(to_librosa(audio))
print(len(normalized))
print(len(y))
print(normalized)
print(y[3000:6000])

logmel, mfcc = extract_features_loaded(normalized, sr)
print(len(np.ndarray.flatten(logmel)))

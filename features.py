import numpy as np
import librosa
import soundfile
from pydub import AudioSegment

from extract_features import to_librosa
from visual_guitar import read_file
from wav_convert import check



#
#
# def cut(segment, start, stop):
#     start = int(start * 1000)
#     stop = int(stop * 1000)
#     return segment[start:stop]
#
#
# name = '/home/moby/PycharmProjects/data/IDMT-SMT-32/dataset2/audio/AR_Lick1_FVSDN.wav'
# name_csv = '/home/moby/PycharmProjects/data/IDMT-SMT-GUITAR_V2/dataset2/csv/AR_Lick1_FVSDN.csv'
# first = read_file(name_csv)[0]
# segment = AudioSegment.from_file(name, format="wav")
# segm = segment[0:5000]
# segm.export('tmp.wav', format='wav')
# libr = to_librosa(segm)
# soundfile.write('tmp2.wav', libr, samplerate=44100, subtype="FLOAT")
# hop_length = 1024
# f1, sr = librosa.load('tmp.wav', sr=44100)
# f2, sr2 = librosa.load('tmp2.wav', sr=44100)
# print(librosa.feature.mfcc(libr, sr=sr, hop_length=hop_length, n_mfcc=1))
# print(librosa.feature.mfcc(f2, sr=sr2, hop_length=hop_length, n_mfcc=1))


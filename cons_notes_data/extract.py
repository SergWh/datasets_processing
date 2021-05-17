import librosa
import librosa.display
import numpy as np

hop_length = 512
win_length = 2048
sample_rate = 44100

gpt_path4 = '/home/moby/PycharmProjects/data/gpt/wav_split/1/slide_half_step_up/slide_half_step_up_2_1.wav'


def extract_features(filename):
    y, sr = librosa.load(filename, sr=sample_rate)
    print(len(y))
    return extract_features_loaded(y, sr)


def extract_features_loaded(y, sr):
    D = np.abs(librosa.stft(y, hop_length=hop_length, win_length=win_length))
    D2 = np.abs(D) ** 2
    S = librosa.feature.melspectrogram(S=D2, sr=sr)
    MFCC = librosa.feature.mfcc(S=librosa.power_to_db(S))
    return S, MFCC


# logmel, mfcc = extract_features(gpt_path4)
# print(len(np.ndarray.flatten(logmel)))

import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.lines import Line2D
from scipy.stats import stats

from note_estimation.common import load_file, SAMPLE_RATE


def extract_melspec(D2):
    return librosa.feature.melspectrogram(S=D2, sr=SAMPLE_RATE)


def extract_spec(y, hop_length=512, win_length=2048):
    D = librosa.stft(y, hop_length=hop_length, win_length=win_length)
    return np.abs(D) ** 2


filename = '/home/moby/PycharmProjects/data/gpt/wav_split/1/bending_up_half/bending_up_half_1_1.wav'
y = load_file(filename)
print(y.shape)
melsp = extract_melspec(extract_spec(y))
print(melsp.shape)
fig, ax = plt.subplots()
S_dB = librosa.power_to_db(melsp, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                               y_axis='mel', sr=SAMPLE_RATE,
                               fmax=8000, ax=ax)
ax.set(xlabel='Время, с')
ax.set(ylabel='Частота, Гц')
plt.show()

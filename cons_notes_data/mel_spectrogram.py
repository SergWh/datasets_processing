import librosa.display
import numpy as np
import matplotlib.pyplot as plt

wav_name = '/home/moby/PycharmProjects/data/IDMT-SMT-32/dataset2/audio/AR_Lick1_FVSDN.wav'
gpt_path = '/home/moby/PycharmProjects/data/gpt/wav_split/1/normal_half_step_up/normal_half_step_up_1_1.wav'
gpt_path2 = '/home/moby/PycharmProjects/data/gpt/wav_split/1/bending_up_half/bending_up_half_1_1.wav'
gpt_path3 = '/home/moby/PycharmProjects/data/gpt/wav_split/1/slide_half_step_up/slide_half_step_up_1_1.wav'
gpt_path4 = '/home/moby/PycharmProjects/data/gpt/wav_split/1/slide_half_step_up/slide_half_step_up_2_1.wav'
sr = 44100

hop_length = 512
win_length = 2048

y, sr = librosa.load(wav_name, sr=sr)

D = np.abs(librosa.stft(y, hop_length=hop_length, win_length=win_length))
db = librosa.amplitude_to_db(D, ref=np.max)
D2 = np.abs(D) ** 2
S = librosa.feature.melspectrogram(S=D2, sr=sr)
MFCC = librosa.feature.mfcc(S=librosa.power_to_db(S))
print(D.shape)
print(S.shape)
print(MFCC.shape)
fig, ax = plt.subplots(nrows=3, sharex=True, sharey=False)
librosa.display.specshow(db, x_axis='time', y_axis='log', ax=ax[0])
ax[0].set(title='Power spectrogram')
ax[0].label_outer()
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), x_axis='time', y_axis='log', ax=ax[0])
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                         x_axis='time',
                         y_axis='mel',
                         fmax=8000, ax=ax[1])
img = librosa.display.specshow(MFCC, x_axis='time', ax=ax[2])
plt.show()


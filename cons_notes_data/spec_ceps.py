import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# wav_name = '/home/moby/PycharmProjects/data/IDMT-SMT-32/dataset2/audio/AR_Lick1_FVSDN.wav'
gpt_path = '/home/moby/PycharmProjects/data/gpt/wav_split/1/normal_half_step_up/normal_half_step_up_1_1.wav'
gpt_path2 = '/home/moby/PycharmProjects/data/gpt/wav_split/1/bending_up_half/bending_up_half_1_1.wav'
sr = 44100


def cepstrum(y):
    return np.fft.ifft(np.log(y))


def st_cepstrum(amp_spec):
    return np.apply_along_axis(cepstrum, 0, amp_spec)


high = librosa.note_to_hz('E4')
low = librosa.note_to_hz('E2')
hop_length = 512
win_length = 2048

y, sr = librosa.load(gpt_path, sr=sr)

D = np.abs(librosa.stft(y, hop_length=hop_length, win_length=win_length))
db = librosa.amplitude_to_db(D, ref=np.max)

ceps = st_cepstrum(D)

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=False)
librosa.display.specshow(db, x_axis='time', y_axis='log', ax=ax[0])
ax[0].set(title='Power spectrogram')
ax[0].label_outer()
librosa.display.specshow(librosa.amplitude_to_db(ceps), x_axis='time', y_axis='log', ax=ax[1])
# librosa.display.specshow(librosa.stft(ceps2), x_axis='time', y_axis='log', ax=ax[2])
ax[1].set_ylim(0, 100)

# onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
# o_env = librosa.onset.onset_strength(y, sr=sr)
# times = librosa.times_like(o_env, sr=sr)
# onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
#
#
# f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=low, fmax=high)
# times_pitch = librosa.times_like(f0)
# ax[1].plot(times, o_env, label='Onset strength')
# ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9, linestyle='--', label='Onsets')
# ax[1].legend()

# ax[2].plot(times_pitch, f0, label='f0', color='red', linewidth=3)

plt.show()

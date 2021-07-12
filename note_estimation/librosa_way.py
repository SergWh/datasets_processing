import librosa
import librosa.display
import numpy as np

from note_estimation.common import SAMPLE_RATE


def get_onsets_librosa(y):
    o_env = librosa.onset.onset_strength(y, sr=SAMPLE_RATE)
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=SAMPLE_RATE, units="time")
    return onset_frames


def get_pitch_librosa(y):
    f0, voiced, prob = librosa.pyin(y, 80, 1200, sr=SAMPLE_RATE)
    return np.rint(librosa.core.hz_to_midi(f0))

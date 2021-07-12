import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.lines import Line2D
from scipy.stats import stats

from note_estimation.note_utils import Style, Note

SAMPLE_RATE = 44100
THRESHOLD = 0.2

wav_name = '/home/moby/PycharmProjects/data/IDMT-SMT-32/dataset2/audio/AR_Lick2_FVSH.wav'


def get_spec(y):
    return np.abs(librosa.stft(y))


def filter_close(onsets, threshold=THRESHOLD):
    vals = []
    last = -100

    def app(_x):
        vals.append(_x)
        nonlocal last
        last = x

    for i, x in enumerate(onsets):
        if i == 0:
            vals.append(x)
            last = x
        else:
            if x - last >= threshold:
                app(x)
    return np.asarray(vals)


def cut_segment(y, start, end):
    (start, end) = librosa.time_to_samples((start, end), sr=SAMPLE_RATE)
    return y[start, end]


def load_file(filename):
    y, sr = librosa.load(filename, sr=SAMPLE_RATE)
    return y


def test_plot_data(signal, ons_fun, pitch_fun):
    onsets = ons_fun(signal)
    filtered_onsets = filter_close(onsets)
    pitches = pitch_fun(signal)
    times = librosa.times_like(pitches, sr=SAMPLE_RATE)

    fig, ax = plt.subplots(nrows=3, sharex=True)
    librosa.display.specshow(librosa.amplitude_to_db(get_spec(signal), ref=np.max), x_axis='time', y_axis='log',
                             ax=ax[0],
                             sr=SAMPLE_RATE)
    ax[0].set(title='Power spectrogram')
    ax[0].label_outer()
    for onset in onsets:
        ax[1].axvline(x=onset, color='orange')
    for onset in filtered_onsets:
        ax[1].axvline(x=onset, color='red')
    ax[2].plot(times, pitches, label='Pitches', color='black', linewidth=3)
    ax[2].legend()
    librosa.display.waveplot(signal, sr=SAMPLE_RATE, ax=ax[1])
    plt.show()


def notes_to_arrays(signal, notes):
    ons_array = np.zeros(signal.shape)
    ons_array[:] = np.NaN


def mode(x):
    mode_info = stats.mode(x, nan_policy='propagate')
    return mode_info[0][0]


def merge_notes(pitches, onsets, end_ind):
    onset_samples = librosa.time_to_samples(onsets, sr=SAMPLE_RATE)
    shifted = np.append(onset_samples[1:], end_ind)
    note_borders = np.column_stack((onset_samples, shifted))
    pitch_samples = librosa.samples_like(pitches)

    def compute(f):
        return mode(np.take(pitches, np.where((pitch_samples >= f[0]) & (pitch_samples < f[1]))[0]))

    pitched_onsets = np.apply_along_axis(compute, 1, note_borders)
    end_time = librosa.samples_to_time(np.array(end_ind), sr=SAMPLE_RATE)
    onsets_shifted = np.append(onsets[1:], end_time)
    res = np.column_stack((onsets, onsets_shifted, pitched_onsets))

    fun = lambda t: Note(start=t[0], end=t[1], style=Style.NORMAL, pitch=t[2])
    return np.apply_along_axis(fun, 1, res)


def plot_notes(signal, notes, plot_nan_pitch=False):
    fig, ax = plt.subplots(nrows=3, sharex=True, gridspec_kw={'height_ratios': [2, 1, 2]})
    librosa.display.specshow(librosa.amplitude_to_db(get_spec(signal), ref=np.max), x_axis='time', y_axis='log',
                             ax=ax[0],
                             sr=SAMPLE_RATE)
    ax[0].label_outer()
    for note in notes:
        if np.isnan(note.pitch) and not plot_nan_pitch:
            continue
        ax[1].axvline(x=note.start, color='black', linewidth=1)
        ax[2].axvspan(note.start, note.end, alpha=0.8, color=get_color(note.style))
        ax[2].hlines(y=note.pitch, xmin=note.start, xmax=note.end, colors='black', linewidth=3)
        ax[2].axvline(x=note.start, color='black', linewidth=1)
    librosa.display.waveplot(signal, sr=SAMPLE_RATE, ax=ax[1])
    ax[2].set(title='Pitch and Style', xlabel='Time', ylabel='Pitch (MIDI)')
    ax[1].set(title='Waveform and offsets', xlabel='', ylabel='Amplitude')
    ax[0].set(title='Power spectrogram')
    ax[1].axes.get_yaxis().set_ticks([])
    fig.tight_layout()
    fig.legend(handles=color_legend(), loc=4)
    fig.subplots_adjust(right=0.75)
    plt.show()


colors = ('red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'white')


def get_color(style):
    return colors[style.value - 1]


def color_legend():
    return list(map(lambda style: Line2D([0], [0], color=get_color(style), alpha=0.8, lw=4, label=style.name), Style))


# def notes_to_samples(sample_num, notes):


'''
audio - signal
onsets - array of s
pitches - array of frames
'''
# def plot_data(audio, onsets, pitches):
# values, counts = np.unique(x, return_counts=True)
# m = counts.argmax()

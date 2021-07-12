import librosa
import librosa.display
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

from note_estimation.common import load_file, get_spec, SAMPLE_RATE

normal = '/home/moby/PycharmProjects/data/gpt/wav_split/1/normal_half_step_up/normal_half_step_up_1_1.wav'
mute = '/home/moby/PycharmProjects/data/gpt/wav_split/1/mute/mute_1_1.wav'
vibrato = '/home/moby/PycharmProjects/data/gpt/wav_split/1/trill/trill_2_1.wav'
hammer = '/home/moby/PycharmProjects/data/gpt/wav_split/1/hamming_half_step/hamming_half_step_1_1.wav'
pull = '/home/moby/PycharmProjects/data/gpt/wav_split/1/pulling_half_step/pulling_half_step_1_1.wav'
slide = '/home/moby/PycharmProjects/data/gpt/wav_split/1/slide_half_step_up/slide_half_step_up_1_1.wav'
bend = '/home/moby/PycharmProjects/data/gpt/wav_split/1/bending_up_half/bending_up_half_1_1.wav'

one_note = (normal, mute, vibrato)
one_note_names = ('normal', 'mute', 'vibrato')
two_notes = (hammer, pull, slide, bend)
two_notes_names = ('Hammer-on', 'Pull-off', 'Slide', 'Bend')

two_notes = (hammer, slide)
two_notes_names = ('Hammer-on', 'Slide', 'Bend')

def plot_spec(filename, ax):
    signal = load_file(filename)
    librosa.display.specshow(librosa.power_to_db(librosa.feature.melspectrogram(signal, sr=SAMPLE_RATE), ref=np.max),
                             x_axis='time',
                             y_axis='mel',
                             ax=ax,
                             sr=SAMPLE_RATE)
    ax.set(xlabel='Время, с')
    ax.set(ylim=(0, 4096))
    ax.set(ylabel='Частота, Гц')


plt.rcParams.update({
    'axes.titlesize': 20,
    'axes.labelsize': 15,
    'font.family': 'serif',
    'font.sans-serif': ['Times'],
    'text.latex.preamble':
        r'\usepackage[T2A]{fontenc}'
        r'\usepackage[utf8]{inputenc}'
})
img_path = '/home/moby/PycharmProjects/datasets_processing/figures/'

fig, ax = plt.subplots(2, 1,
                       figsize=(6, 9)
                       )
print(ax)
for i, n in enumerate(two_notes):
    plot_spec(n, ax[i])
    ax[i].set_title(two_notes_names[i])
# for i, n in enumerate(one_note):
#     plot_spec(n, ax[i])
#     ax[i].set_title(one_note_names[i])

fig.tight_layout()
# ax[1, 3].set_visible(False)

plt.savefig(img_path + "multi_spec_vert" + '.pdf')
plt.show()

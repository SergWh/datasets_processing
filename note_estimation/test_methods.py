import librosa
import numpy as np

from note_estimation.common import load_file, wav_name, filter_close, merge_notes, SAMPLE_RATE
from note_estimation.dataset_annotations import read_notes
from note_estimation.librosa_way import get_pitch_librosa, get_onsets_librosa
from note_estimation.metrics import calculate_metrics


def sub_array(n, extract_fun):
    (start, end) = librosa.time_to_samples((n.start, n.end), sr=SAMPLE_RATE)
    return np.full(end - start, extract_fun(n))


def create_sample_metric_array(audio, notes, extract_fun, def_val=np.NaN):
    res = np.full(audio.shape, def_val)

    def sub_ar(n):
        (start, end) = librosa.time_to_samples((n.start, n.end), sr=SAMPLE_RATE)
        return range(start, end), extract_fun(n)

    for n in notes:
        inds, val = sub_ar(n)
        res.put(inds, val, mode='clip')

    return res


audio = load_file(wav_name)
pitches = (get_pitch_librosa(audio))
onsets = (get_onsets_librosa(audio))
f = filter_close(onsets)
notes = merge_notes(pitches, f, audio.shape[0] - 1)
bla = lambda n: n.pitch

csv_name = '/home/moby/PycharmProjects/data/IDMT-SMT-32/dataset2/csv/AR_Lick2_FVSH.csv'
dataset_notes = read_notes(csv_name)
pitch_samples_librosa = create_sample_metric_array(audio, notes, bla)
pitch_samples_annotation = create_sample_metric_array(audio, dataset_notes, bla)

# print(calculate_metrics(pitch_samples_annotation, pitch_samples_librosa))

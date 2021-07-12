import glob

import numpy as np

from note_estimation.common import load_file
from note_estimation.dataset_annotations import read_notes, read_notes_non_np
from note_estimation.metrics import calculate_metrics
from note_estimation.test_methods import create_sample_metric_array


def get_pitch(note):
    return note.pitch


def get_style(note):
    return note.style.value


audio_files = sorted(glob.glob("/home/moby/PycharmProjects/data/IDMT-SMT-32/dataset2/audio/*.wav"))
audios = list(map(load_file, audio_files))


def read_trans(path):
    files = sorted(glob.glob(path + "*.csv"))
    read = lambda file:  [x for x in read_notes_non_np(file) if x.pitch != -1]
    trans = list(map(read, files))
    return trans


def notes_to_samples_pitch(trans):
    create_array = lambda p: create_sample_metric_array(audios[p[0]], p[1], get_pitch)
    res = (list(map(create_array, enumerate(trans))))
    return res


def compare_sets(anns, estims):
    res = []
    for i, n in enumerate(estims):
        res.append(calculate_metrics(np.asarray(anns[i]), np.asarray(n)))
    return np.average(np.asarray(res), axis=0)


# onf = notes_to_samples_pitch(read_trans('/home/moby/PycharmProjects/data/onf_notes/'))
alg = notes_to_samples_pitch(read_trans('/home/moby/PycharmProjects/data/alg/'))
anns = notes_to_samples_pitch(read_trans('/home/moby/PycharmProjects/data/IDMT-SMT-32/dataset2/csv/'))

# print("ONF")
# print(compare_sets(anns, onf))
# print()
print("ALG")
print(compare_sets(anns, alg))

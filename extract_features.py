import csv
import glob
import os

from utils import Note, Style
from visual_guitar import read_file
import numpy as np
import librosa
import soundfile
from pydub import AudioSegment
from scipy.stats import kurtosis, skew

from visual_guitar import read_file
from wav_convert import check


def get_start_end_ms(note):
    return int(note.start * 1000), int(note.end * 1000)


def extract_by_ms(audio, start, end):
    return audio[start * 1000:end * 1000]


def extract_by_note(audio, note):
    start, end = get_start_end_ms(note)
    return audio[start:end]


def get_segments(wav_name, csv_name):
    notes = read_file(csv_name)
    audio = AudioSegment.from_file(wav_name, format="wav")
    return list(map(lambda n: extract_by_note(audio, n), notes))


def get_segments_with_pitch(wav_name, csv_name):
    notes = read_file(csv_name)
    audio = AudioSegment.from_file(wav_name, format="wav")
    return list(map(lambda n: (extract_by_note(audio, n), n.pitch), notes))


def to_librosa(audiosegment):
    channel_sounds = audiosegment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    fp_arr = fp_arr.reshape(-1)

    return fp_arr


name = '/home/moby/PycharmProjects/data/IDMT-SMT-32/dataset2/audio/AR_Lick1_FVSDN.wav'
name_csv = '/home/moby/PycharmProjects/data/IDMT-SMT-GUITAR_V2/dataset2/csv/AR_Lick1_FVSDN.csv'

sr = 44100
hop_length = 512
frame_length = 1024


def extract_stft(array, hop_length=hop_length, win_length=frame_length):
    return librosa.stft(array, hop_length=hop_length, win_length=win_length)


def extract_mel(stft):
    stft_2 = np.abs(stft) ** 2
    return librosa.feature.melspectrogram(S=stft_2, sr=sr)


def extract_mfcc(mel):
    return librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=13)


def extract_centroid(stft):
    s, ph = librosa.magphase(stft)
    return librosa.feature.spectral_centroid(S=s)


def extract_zcr(audio):
    return librosa.feature.zero_crossing_rate(audio, frame_length=frame_length, hop_length=hop_length)


def extract_flatness(stft):
    s, ph = librosa.magphase(stft)
    return librosa.feature.spectral_flatness(S=s)


def compute_stats(array):
    funs = [np.max, np.min, np.average, np.mean, np.median, np.var, kurtosis, skew]
    return list(map(lambda f: np.apply_along_axis(f, 1, array), funs))


flatten = lambda t: [item for sublist in t for item in sublist]


def flat_list(list_of_lists):
    return [y for x in list_of_lists for y in x]


def extract_vector(segment):
    librosa_arr = to_librosa(segment)
    stft = extract_stft(librosa_arr)
    mels = extract_mel(stft)

    mfccs = extract_mfcc(mels)
    centr = extract_centroid(stft)
    zcr = extract_zcr(librosa_arr)
    flatness = extract_flatness(stft)

    features = [mfccs, centr, zcr, flatness]
    list_of_stats = (map(compute_stats, features))
    return flat_list(flat_list(list_of_stats))


def get_notes_with_vectors(wav_name, csv_name):
    notes = read_file(csv_name)
    audio = AudioSegment.from_file(wav_name, format="wav")
    return list(map(lambda n: (n, extract_vector(extract_by_note(audio, n))), notes))


# def rewrite_csv(filename, n, vector):
#     paths = filename.split('/')
#     new_folder = 'IDMT-SMT-32'
#     a = paths[:len(paths) - n - 1]
#     a.append(new_folder)
#     a.extend(paths[-n:])
#     new_name = '/'.join(a)
#     print(new_name)

def rewrite_csv(name, pairs):
    header = ['start', 'end', 'pitch', 'velocity', 'style']
    os.makedirs(os.path.dirname(name), exist_ok=True)
    with open(name, 'w+') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
        for (note, features) in pairs:
            arr = [note.start, note.end, note.pitch, note.velocity, note.style.to_str()]
            arr.extend(features)
            writer.writerow(arr)


def read_floats(lst):
    return [float(i) for i in lst]


def read_csv(name):
    with open(name, 'r') as csvfile:
        rest = 'rest'
        reader = csv.DictReader(csvfile, restkey=rest)
        return list(map(lambda r: (Note.parse_csv_row(r), read_floats(r[rest])), reader))


idmt_path = '/home/moby/PycharmProjects/data/IDMT-SMT-32/'
gpt_path = '/home/moby/PycharmProjects/data/gpt/'
templ_idmt_1 = '/*/*/csv/*.csv'
templ_idmt_2 = '/*/csv/*.csv'
gpt_templ = 'csv/*/*/*.csv'
folder_name = 'csv2'


def rewrite_idmt(name, template):
    files = glob.glob(name + template, recursive=True)
    for csv_name in files[:1]:
        paths = csv_name.split('.')[0].split('/')
        first = paths[:len(paths) - 2]
        middle = ['audio']
        second = paths[-1:]
        wav_paths = (first + middle + second)
        wav_name = '/'.join(wav_paths) + '.wav'
        pairs = get_notes_with_vectors(wav_name, csv_name)
        new_csv_paths = paths[:len(paths) - 2] + [folder_name] + paths[-1:]
        new_csv_name = '/'.join(new_csv_paths) + '.csv'
        rewrite_csv(new_csv_name, pairs)


def rewrite_gpt(name, template):
    files = glob.glob(name + template, recursive=True)
    for csv_name in files:
        paths = csv_name.split('.')[0].split('/')
        first = paths[:len(paths) - 4]
        middle = ['wav_split']
        second = paths[-3:]
        wav_paths = (first + middle + second)
        wav_name = '/'.join(wav_paths) + '.wav'
        pairs = get_notes_with_vectors(wav_name, csv_name)
        new_csv_paths = paths[:len(paths) - 4] + [folder_name] + paths[-3:]
        new_csv_name = '/'.join(new_csv_paths) + '.csv'
        rewrite_csv(new_csv_name, pairs)


rewrite_gpt(gpt_path, gpt_templ)

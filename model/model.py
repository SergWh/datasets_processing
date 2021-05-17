import numpy as np
from pydub import AudioSegment
import librosa


def get_segment(audio, start, end):
    return audio[int(start * 1000): int(end * 1000)]


def to_librosa(audiosegment):
    channel_sounds = audiosegment.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    fp_arr = fp_arr.reshape(-1)

    return fp_arr


def normalize(librosa_samples, middle=None, max=13231):
    # if middle is not None:
    #     return cut(librosa_samples, middle)
    if len(librosa_samples) > max:
        return cut(librosa_samples, middle)
    else:
        return librosa.util.pad_center(librosa_samples, max)


# cut
def cut(librosa_samples, middle=None, max_l=13231):
    if middle is not None:
        mid = librosa.time_to_samples(middle, sr=44100)
        l_half = max_l // 2
        l_ind = (max(0, mid - l_half))
        r_ind = l_ind + max_l
        return librosa.util.pad_center(librosa_samples[l_ind: r_ind], max_l)
    else:
        return librosa.util.fix_length(librosa_samples[:max_l + 1], max_l)


def prepare_segment(audio, start, end, middle=None, max_l=13231):
    return normalize(to_librosa(get_segment(audio, start, end)), middle=middle, max=max_l)


def extract_segments(notes, file_name):
    if len(notes) == 0:
        return

    notes.sort(key=lambda note: note.start_time)
    pairs = list(zip(notes, notes[1:]))
    first = notes[0]
    audio = AudioSegment.from_file(file_name, format="wav")
    first_segm = prepare_segment(audio, first.start_time, first.end_time)
    other = map(lambda t: prepare_segment(audio, t[0].start_time, t[1].end_time, middle=t[1].start_time), pairs)
    return [first_segm].extend(other)

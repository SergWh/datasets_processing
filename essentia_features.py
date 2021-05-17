import librosa

from extract_features import get_segments, to_librosa, extract_stft, get_segments_with_pitch
import essentia.standard

name = '/home/moby/PycharmProjects/data/IDMT-SMT-32/dataset2/audio/AR_Lick1_FVSDN.wav'
name_csv = '/home/moby/PycharmProjects/data/IDMT-SMT-GUITAR_V2/dataset2/csv/AR_Lick1_FVSDN.csv'

segments = get_segments_with_pitch(name, name_csv)
segm, pitch = segments[4]
librosa_arr = to_librosa(segm)
stft = extract_stft(librosa_arr)
magnitudes, ph = librosa.magphase(stft)

peaks = essentia.standard.SpectralPeaks()
freqs, mags = peaks(magnitudes[0])
print(freqs)

# pitch = essentia.standard.PitchYin()
# pitch, conf = pitch(librosa_arr)
print(pitch)
harm_peaks = essentia.standard.HarmonicPeaks()
harm_freqs, harm_mags = harm_peaks(freqs, mags, pitch)

inharmonicity = essentia.standard.Inharmonicity()
inharm = inharmonicity(harm_freqs, harm_mags)

tristimulus = essentia.standard.Tristimulus()
trist = tristimulus(harm_freqs, harm_mags)

print(inharm)
print()
print(trist)

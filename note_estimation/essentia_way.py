import essentia
import essentia.standard as es
import librosa
import numpy as np


def get_onsets_essentia(audio):
    od1 = es.OnsetDetection(method='complex_phase')

    w = es.Windowing(type='hann')
    fft = es.FFT()
    c2p = es.CartesianToPolar()
    pool = essentia.Pool()
    for frame in es.FrameGenerator(audio, frameSize=1024, hopSize=512):
        mag, phase, = c2p(fft(w(frame)))
        pool.add('features.complex', od1(mag, phase))

    onsets = es.Onsets()
    return onsets(essentia.array([pool['features.complex']]), [1])


def get_pitch_essentia(audio):
    mel = es.PredominantPitchMelodia()
    pitch, conf = mel(audio)
    ints = np.rint(librosa.core.hz_to_midi(pitch))
    ints[ints == -np.inf] = np.NaN
    ints[ints == np.inf] = np.NaN
    return ints

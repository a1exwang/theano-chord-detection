import scipy.io.wavfile
import scipy.signal
import wave
import numpy as np
import numpy.fft as fft
import librosa
from maps_db import PIANO_KEY_COUNT, HALVES_PER_OCTAVE, FIRST_PIANO_KEY_MIDI_VALUE


def play(model, file_path, freq_count, count_bins, duration):
    start_time = 0.5

    (sample_rate, data) = scipy.io.wavfile.read(file_path, mmap=True)
    # Merge into one channel
    start_index = int(start_time * sample_rate)
    length = int(duration * sample_rate)

    wav = wave.open(file_path, 'r')
    # Max value for n-bit signed integers.
    max_value = float(2 ** (8 * wav.getsampwidth() - 1))
    wav.close()

    normalized_data = data / max_value
    part = np.sum(normalized_data[start_index:(start_index + length)], axis=1)

    # DFT
    resampled_part = scipy.signal.resample(
        part,
        freq_count * 2,
        window=np.hanning(length))

    fft_freqs = fft.fft(resampled_part)
    fft_freqs = np.abs(fft_freqs[0:freq_count])
    fft_freqs = np.reshape(fft_freqs, [1, freq_count])
    current_dft_freqs = fft_freqs

    # CQT
    hop_length = 16384
    midis = librosa.cqt(part,
                        sr=sample_rate,
                        hop_length=hop_length,
                        fmin=librosa.midi_to_hz(FIRST_PIANO_KEY_MIDI_VALUE),
                        bins_per_octave=count_bins / PIANO_KEY_COUNT * HALVES_PER_OCTAVE,
                        n_bins=count_bins,
                        real=True)
    current_cqt_freqs = np.reshape(midis[:, 0], [1, count_bins])

    vec = np.append(current_cqt_freqs, current_dft_freqs, axis=1)
    one_hot = model.predict(vec)
    key = np.argmax(one_hot)
    return {'output': one_hot, 'piano_key': key, 'max_value': one_hot[0][0][key]}

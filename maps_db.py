import numpy as np
import numpy.fft as fft
import scipy.io.wavfile
import scipy.signal
import wave
import re
import os
import librosa
import stat
import redis
import pickle
import random
from ast import literal_eval

FIRST_PIANO_KEY_MIDI_VALUE = 21
FIRST_PIANO_KEY_FREQUENCY = 17.5
PIANO_KEY_COUNT = 88
HALVES_PER_OCTAVE = 12


class MapsDB:
    class Sample:
        def __init__(self, cqt_freqs, dft_freqs, batch_piano_keys, batch_size):
            self.cqt_freqs = cqt_freqs
            self.dft_freqs = dft_freqs
            self.one_hot_piano_keys = np.zeros([batch_size, PIANO_KEY_COUNT])
            self.piano_keys = batch_piano_keys
            for index, piano_keys in enumerate(batch_piano_keys):
                for key in piano_keys:
                    self.one_hot_piano_keys[index, key] = 1

        def vec_input(self):
            if self.dft_freqs is None:
                vec = self.cqt_freqs
            else:
                vec = np.append(self.cqt_freqs, self.dft_freqs, axis=1)
            return vec

        def label(self):
            return self.one_hot_piano_keys

    def __init__(self, dir_path, type1='ISOL', type2='NO', freq_count=10000,
                 start_time=0.5, duration=0.5, count_bins=88, shuffle=True, use_dft=True, batch_size=4):
        self.rserver = redis.Redis()
        self.dir_path = dir_path
        self.freq_count = freq_count
        self.start_time = start_time
        self.duration = duration
        self.count_bins = count_bins
        self.cached_samples = {}
        self.use_dft = use_dft
        self.batch_size = batch_size
        self.shuffle = shuffle
        wav_files = []

        for instr_name in os.listdir(dir_path):
            instr_dir = os.path.join(dir_path, instr_name)
            mode = os.stat(instr_dir)[stat.ST_MODE]
            if not stat.S_ISDIR(mode):
                continue
            type_dirs = os.listdir(instr_dir)
            if type1 in type_dirs:
                full_path = os.path.join(dir_path, instr_name, type1, type2)
                mode = os.stat(full_path)[stat.ST_MODE]
                if not stat.S_ISDIR(mode):
                    continue

                for file_name in os.listdir(full_path):
                    if file_name.endswith('.wav'):
                        # MAPS_ISOL_NO_[loudness]_S[pedal_pressed]_M[midi]_[instrument].wav
                        m = re.match(r'MAPS_ISOL_NO_(\w)_S(\d)_M(\d+)_' + instr_name + '\.wav', file_name)
                        if not m:
                            raise RuntimeError("Unknown Data")
                        loudness = m.group(1)
                        pedal_pressed = int(m.group(2))
                        piano_key = int(m.group(3)) - FIRST_PIANO_KEY_MIDI_VALUE

                        wav_files.append(
                            (piano_key,  full_path + '/' + file_name))
        wav_files.sort(key=lambda tup: tup[1])
        # take even numbered files as train data
        self.train_data = wav_files
        self.test_data = wav_files[0::30]

    def load_cache(self):
        self.deserialize_samples()

    def get_vec_input_width(self):
        if self.use_dft:
            return self.freq_count + self.count_bins
        else:
            return self.count_bins

    @staticmethod
    def get_label_width():
        return PIANO_KEY_COUNT

    def train_iterator(self):
        for val in self.data_iterator(self.train_data):
            yield val

    def test_iterator(self):
        for val in self.data_iterator(self.test_data):
            yield val

    def data_iterator(self, wav_files):
        current_batch_index = 0
        if self.use_dft:
            dft_freqs = np.zeros([self.batch_size, self.freq_count], dtype='float32')
        else:
            dft_freqs = None

        cqt_freqs = np.zeros([self.batch_size, self.count_bins], dtype='float32')
        piano_keys = [[]] * self.batch_size

        random.shuffle(wav_files)
        for (piano_key, file_path) in wav_files:
            if file_path in self.cached_samples:
                current_cqt_freqs, current_dft_freqs, current_piano_keys = \
                    self.cached_samples[file_path]
            else:
                (sample_rate, data) = scipy.io.wavfile.read(file_path, mmap=True)
                # Merge into one channel
                start_index = int(self.start_time * sample_rate)
                length = int(self.duration * sample_rate)

                wav = wave.open(file_path, 'r')
                # Max value for n-bit signed integers.
                max_value = float(2 ** (8 * wav.getsampwidth() - 1))
                wav.close()

                normalized_data = data / max_value
                part = np.sum(normalized_data[start_index:(start_index + length)], axis=1)

                # DFT
                if self.use_dft:
                    resampled_part = scipy.signal.resample(
                        part,
                        self.freq_count * 2,
                        window=np.hanning(length))

                    fft_freqs = fft.fft(resampled_part)
                    fft_freqs = np.abs(fft_freqs[0:self.freq_count])
                    fft_freqs = np.reshape(fft_freqs, [1, self.freq_count])
                    current_dft_freqs = fft_freqs
                else:
                    current_dft_freqs = None

                # CQT
                hop_length = 16384
                midis = librosa.cqt(part,
                                    sr=sample_rate,
                                    hop_length=hop_length,
                                    fmin=librosa.midi_to_hz(FIRST_PIANO_KEY_MIDI_VALUE),
                                    bins_per_octave=self.count_bins / PIANO_KEY_COUNT * HALVES_PER_OCTAVE,
                                    n_bins=self.count_bins,
                                    real=True)
                current_cqt_freqs = np.reshape(midis[:, 0], [1, self.count_bins])
                # Piano key
                current_piano_keys = [piano_key]

                self.cached_samples[file_path] = (current_cqt_freqs, current_dft_freqs, current_piano_keys)
                self.serialize_sample(file_path, current_cqt_freqs, current_dft_freqs, current_piano_keys)

            cqt_freqs[current_batch_index, :] = current_cqt_freqs
            if self.use_dft:
                dft_freqs[current_batch_index, :] = current_dft_freqs
            piano_keys[current_batch_index] = current_piano_keys

            current_batch_index += 1
            if current_batch_index == self.batch_size:
                current_batch_index = 0
                yield MapsDB.Sample(cqt_freqs, dft_freqs, piano_keys, self.batch_size)

    def flush_db_if_too_old(self):
        if self.rserver.get('maps_db_guid') != self.redis_guid():
            self.rserver.flushdb()
            self.rserver.set('maps_db_guid', self.redis_guid())
            return True
        return False

    def redis_guid(self):
        return "guid_%d_%f_%f_%d_%d_10" % \
               (self.freq_count, self.start_time, self.duration, self.count_bins, self.use_dft)

    def serialize_sample(self, filename, cqt, dft, keys):
        self.rserver.set(filename,  (pickle.dumps(cqt), pickle.dumps(dft), pickle.dumps(keys)))

    def deserialize_samples(self):
        if not self.flush_db_if_too_old():
            # deserialize
            for file_path in self.rserver.scan_iter("..*"):
                a = self.rserver.get(file_path)
                (cqt_, dft_, keys_) = literal_eval(a)
                cqt = pickle.loads(cqt_)
                dft = pickle.loads(dft_)
                keys = pickle.loads(keys_)
                self.cached_samples[file_path] = (cqt, dft, keys)

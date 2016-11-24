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


class MapsFileNameGen:
    MAPS_AUDIO_FILE_SUFFIX = '.wav'

    def __init__(self, db_dir):
        self.db_path = db_dir

    @staticmethod
    def _is_dir(path):
        return os.path.isdir(path)

    def isol_no(self):
        db_name = 'ISOL'
        isol_type = 'NO'
        for instr_name in os.listdir(self.db_path):
            instr_dir = os.path.join(self.db_path, instr_name)
            if not self._is_dir(instr_dir):
                continue
            type_dirs = os.listdir(instr_dir)
            if db_name not in type_dirs:
                continue
            if not self._is_dir(os.path.join(self.db_path, instr_name, db_name)):
                continue
            full_path = os.path.join(self.db_path, instr_name, db_name, isol_type)
            if not self._is_dir(full_path):
                continue
            for file_name in os.listdir(full_path):
                if not file_name.endswith(self.MAPS_AUDIO_FILE_SUFFIX):
                    continue
                # MAPS_ISOL_NO_[loudness]_S[pedal_pressed]_M[midi]_[instrument].wav
                m = re.match(r'MAPS_ISOL_NO_(\w)_S(\d)_M(\d+)_\w+\.wav', file_name)
                if not m:
                    raise RuntimeError("Unknown Data")
                piano_key = int(m.group(3)) - FIRST_PIANO_KEY_MIDI_VALUE

                yield ([piano_key],
                       os.path.join(full_path, file_name))

    def ucho(self, one_octave=True):
        db_name = 'UCHO'
        db_type_one_octave = 'I60-68'
        db_type_full = 'I32-96'
        db_types = [db_type_one_octave] if one_octave else [db_type_one_octave, db_type_full]
        for instr_name in os.listdir(self.db_path):
            instr_dir = os.path.join(self.db_path, instr_name)
            if not self._is_dir(instr_dir):
                continue
            if not self._is_dir(os.path.join(instr_dir, db_name)):
                continue

            for db_type in db_types:
                full_path = os.path.join(instr_dir, db_name, db_type)
                if not self._is_dir(full_path):
                    raise RuntimeError('Directory not found')
                for chord_type in os.listdir(full_path):
                    for file_name in os.listdir(os.path.join(full_path, chord_type)):
                        if not file_name.endswith(self.MAPS_AUDIO_FILE_SUFFIX):
                            continue
                        wav_path = os.path.join(full_path, chord_type, file_name)
                        txt_path = os.path.join(full_path, chord_type, file_name[0:-3] + 'txt')
                        with open(txt_path) as f:
                            piano_keys = map(lambda x: int(x.split()[-1]) - FIRST_PIANO_KEY_MIDI_VALUE,
                                             f.read().split('\n')[1:-1])
                            yield (piano_keys, wav_path)


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

        def label_key_count(self):
            return map(lambda keys: len(keys), self.piano_keys)

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

        # TODO: read wav_files array from redis rather than disk
        wav_files = []
        file_names = MapsFileNameGen(dir_path)
        for piano_keys, full_path in file_names.ucho(True):
            wav_files.append((piano_keys, full_path))
        for piano_keys, full_path in file_names.isol_no():
            wav_files.append((piano_keys, full_path))
        # Sort by file path
        wav_files.sort(key=lambda tup: tup[1])

        # TODO: seperate train data and test data
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
        for (current_piano_keys, file_path) in wav_files:
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

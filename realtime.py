from array import array
from struct import pack
from sys import byteorder
import copy
import pyaudio
import wave
import numpy as np
import time

THRESHOLD = 500  # audio levels not normalised.
DURATION = 0.5
SILENT_CHUNKS = 3 * 44100 / 1024  # about 3sec
FORMAT = pyaudio.paInt16
FRAME_MAX_VALUE = 2 ** 15 - 1
NORMALIZE_MINUS_ONE_dB = 10 ** (-1.0 / 20)
SAMPLE_RATE = 44100
CHUNK_SIZE = int(DURATION * SAMPLE_RATE)
CHANNELS = 1
TRIM_APPEND = SAMPLE_RATE / 4
BUFFERED_SECONDS = 5
BUFFERED_LENGTH = BUFFERED_SECONDS * SAMPLE_RATE
QUEUE_SIZE = 10


def is_silent(data_chunk):
    """Returns 'True' if below the 'silent' threshold"""
    return max(data_chunk) < THRESHOLD


def normalize(data_all):
    """Amplify the volume out to max -1dB"""
    # MAXIMUM = 16384
    normalize_factor = (float(NORMALIZE_MINUS_ONE_dB * FRAME_MAX_VALUE)
                        / max(abs(i) for i in data_all))

    r = array('h')
    for i in data_all:
        r.append(int(i * normalize_factor))
    return r


def trim(data_all):
    _from = 0
    _to = len(data_all) - 1
    for i, b in enumerate(data_all):
        if abs(b) > THRESHOLD:
            _from = max(0, i - TRIM_APPEND)
            break

    for i, b in enumerate(reversed(data_all)):
        if abs(b) > THRESHOLD:
            _to = min(len(data_all) - 1, len(data_all) - 1 - i + TRIM_APPEND)
            break

    return copy.deepcopy(data_all[_from:(_to + 1)])


def streaming():
    """Record a word or words from the microphone and
    return the data as an array of signed shorts."""

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK_SIZE)
    queue = []

    while True:
        # little endian, signed short
        data_chunk = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            data_chunk.byteswap()
        np_data = np.array(data_chunk, dtype='float32')
        if len(queue) >= QUEUE_SIZE:
            del queue[0]
        t = time.time()
        yield (t, np_data)
        queue.append((t, np_data))


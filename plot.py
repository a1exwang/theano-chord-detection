from maps_db import MapsDB
import matplotlib.pyplot as plt
import numpy as np

duration = 0.5
start_time = 0.5
freq_count = 4000
freq_resolution = 1 / duration
bins_per_half = 20
count_bins = 88 * bins_per_half
dataset = MapsDB('../db',
                 freq_count=freq_count,
                 count_bins=count_bins,
                 start_time=start_time,
                 duration=duration)
it = dataset.train_iterator(1)
sample = it.next()
key = sample.piano_keys[0]
cqt_freqs = sample.cqt_freqs
dft_freqs = sample.dft_freqs

midi_values = np.array(range(count_bins)) / bins_per_half + 21
fs = np.array(range(freq_count)) * freq_resolution
print(key + 21)

plt.plot(midi_values, np.reshape(cqt_freqs, count_bins))
plt.ylabel('E')
plt.xlabel('f')
plt.show()

plt.plot(fs, np.reshape(dft_freqs, freq_count))
plt.show()

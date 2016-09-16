"""
"""
from __future__ import division

import fractions
import os

import sacred
from tqdm import tqdm
import numpy as np
import scipy.io.wavfile
import scipy.signal
from picklable_itertools import cycle
from picklable_itertools.extras import partition_all
import warnings


# TODO: make SACRED ingredient.
def one_hot(x):
    return np.eye(256, dtype='uint8')[x.astype('uint8')]


def fragment_indices(full_sequences, fragment_length, batch_size, fragment_stride, nb_output_bins):
    for seq_i, sequence in enumerate(full_sequences):
        # range_values = np.linspace(np.iinfo(sequence.dtype).min, np.iinfo(sequence.dtype).max, nb_output_bins)
        # digitized = np.digitize(sequence, range_values).astype('uint8')
        for i in xrange(0, sequence.shape[0] - fragment_length, fragment_stride):
            yield (seq_i, i, i + fragment_length), (seq_i, i + 1, i + 1 + fragment_length)


def batch_generator(full_sequences, fragment_length, batch_size, fragment_stride, nb_output_bins, learn_all_outputs):
    indices = list(fragment_indices(full_sequences, fragment_length, batch_size, fragment_stride, nb_output_bins))
    # TODO: shuffle
    batches = cycle(partition_all(batch_size, indices))
    for batch in batches:
        if len(batch) < batch_size:
            continue
        yield np.array(
            [one_hot(full_sequences[e[0][0]][e[0][1]:e[0][2]]) for e in batch], dtype = 'uint8'), np.array(
            [one_hot(full_sequences[e[1][0]][e[1][1]:e[1][2]]) for e in batch], dtype='uint8')


def generators(dirname, desired_sample_rate, fragment_length, batch_size, fragment_stride, nb_output_bins,
               learn_all_outputs, use_ulaw):
    fragment_generators = {}
    nb_examples = {}
    for set_name in ['train', 'test']:
        set_dirname = os.path.join(dirname, set_name)
        full_sequences = load_set(desired_sample_rate, set_dirname, use_ulaw)
        fragment_generators[set_name] = batch_generator(full_sequences, fragment_length, batch_size, fragment_stride,
                                                        nb_output_bins, learn_all_outputs)
        nb_examples[set_name] = int(sum(
            [len(xrange(0, x.shape[0] - fragment_length, fragment_stride)) for x in
             full_sequences]) / batch_size) * batch_size

    return fragment_generators, nb_examples


def load_set(desired_sample_rate, set_dirname, use_ulaw):
    ulaw_str = '_ulaw' if use_ulaw else ''
    cache_fn = os.path.join(set_dirname, 'processed_%d%s.npy' % (desired_sample_rate, ulaw_str))
    if os.path.isfile(cache_fn):
        full_sequences = np.load(cache_fn)
    else:
        file_names = [fn for fn in os.listdir(set_dirname) if fn.endswith('.wav')]
        full_sequences = []
        for fn in tqdm(file_names):
            sequence = process_wav(desired_sample_rate, os.path.join(set_dirname, fn), use_ulaw)
            full_sequences.append(sequence)
        np.save(cache_fn, full_sequences)

    return full_sequences


def process_wav(desired_sample_rate, filename, use_ulaw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        channels = scipy.io.wavfile.read(filename)
    file_sample_rate, audio = channels
    audio = ensure_mono(audio)
    audio = wav_to_float(audio)
    if use_ulaw:
        audio = ulaw(audio)
    audio = ensure_sample_rate(desired_sample_rate, file_sample_rate, audio)
    audio = float_to_uint8(audio)
    return audio


def ulaw(x, u=255):
    x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
    return x


def float_to_uint8(x):
    x += 1.
    x /= 2.
    uint8_max_value = np.iinfo('uint8').max
    x *= uint8_max_value
    x = x.astype('uint8')
    return x


def wav_to_float(x):
    try:
        max_value = np.iinfo(x.dtype).max
        min_value = np.iinfo(x.dtype).min
    except:
        max_value = np.finfo(x.dtype).max
        min_value = np.iinfo(x.dtype).min
    x = x.astype('float64', casting='safe')
    x -= min_value
    x /= ((max_value - min_value) / 2.)
    x -= 1.
    return x


def ulaw2lin(x, u=255.):
    max_value = np.iinfo('uint8').max
    min_value = np.iinfo('uint8').min
    x = x.astype('float64', casting='safe')
    x -= min_value
    x /= ((max_value - min_value) / 2.)
    x -= 1.
    x = np.sign(x) * (1 / u) * (((1 + u) ** np.abs(x)) - 1)
    x = float_to_uint8(x)
    return x

def ensure_sample_rate(desired_sample_rate, file_sample_rate, mono_audio):
    if file_sample_rate != desired_sample_rate:
        mono_audio = scipy.signal.resample_poly(mono_audio, desired_sample_rate, file_sample_rate)
        # mono_audio = scipy.signal.resample(mono_audio, int((len(mono_audio)*desired_sample_rate) / file_sample_rate))
        # TODO: listen to output
    return mono_audio


def ensure_mono(raw_audio):
    """
    Just use first channel.
    """
    if raw_audio.ndim == 2:
        raw_audio = raw_audio[:, 0]
    return raw_audio


if __name__ == '__main__':
    fragment_length = 2 ** 4
    desired_sample_rate = 44100
    batch_size = 2
    nb_output_bins = 256
    nb_filters = 256
    nb_dilations = 2
    nb_stacks = 1
    fragment_stride = 44100
    skip_connections = False
    g, n = generators(dirname='data', desired_sample_rate=desired_sample_rate, fragment_length=fragment_length,
                      batch_size=batch_size, fragment_stride=fragment_stride, nb_output_bins=nb_output_bins,
                      use_ulaw=True)
    print g['train'].next()[0].shape
    print g['train'].next()[1].shape
    # print len(list(g['train'])), n['train']

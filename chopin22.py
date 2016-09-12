from __future__ import division

import fractions
import os
from tqdm import tqdm
import numpy as np
import scipy.io.wavfile
import scipy.signal
from picklable_itertools import cycle
from picklable_itertools.extras import partition_all


# TODO: make SACRED ingredient.
def one_hot(x):
    return np.eye(256)[x.astype('uint')]


def fragment_generator(full_sequences, fragment_length, batch_size, fragment_stride, nb_output_bins):
    for sequence in full_sequences:
        range_values = np.linspace(np.iinfo(sequence.dtype).min, np.iinfo(sequence.dtype).max, nb_output_bins)
        digitized = np.digitize(sequence, range_values).astype('uint8')
        for i in xrange(0, sequence.shape[0] - fragment_length, fragment_stride):
            yield (one_hot(digitized[i:i + fragment_length]), one_hot(digitized[i + 1:i + 1 + fragment_length]))


def batch_generator(full_sequences, fragment_length, batch_size, fragment_stride, nb_output_bins):
    # TODO: shuffle
    parts = fragment_generator(full_sequences, fragment_length, batch_size, fragment_stride, nb_output_bins)
    batches = cycle(partition_all(batch_size, parts))
    for batch in batches:
        if len(batch) < batch_size:
            continue
        yield np.array([e[0] for e in batch]), np.array([e[1] for e in batch])


def generators(dirname, desired_sample_rate, fragment_length, batch_size, fragment_stride, nb_output_bins):
    """
    Generator
    :param full_sequences: list of sequences
    :param fragment_length: length of parts in seconds.
    :param desired_sample_rate: sample_rate of sequences.
    :return:
    """
    fragment_generators = {}
    nb_examples = {}
    for set_name in ['train', 'test']:
        set_dirname = os.path.join(dirname, set_name)
        full_sequences = load_set(desired_sample_rate, set_dirname)
        fragment_generators[set_name] = batch_generator(full_sequences, fragment_length, batch_size, fragment_stride,
                                                        nb_output_bins)
        nb_examples[set_name] = int(sum(
            [len(xrange(0, x.shape[0] - fragment_length, fragment_stride)) for x in
             full_sequences]) / batch_size) * batch_size

    return fragment_generators, nb_examples


def load_set(desired_sample_rate, set_dirname):
    cache_fn = os.path.join(set_dirname, 'processed_%d.npy' % desired_sample_rate)
    if os.path.isfile(cache_fn):
        full_sequences = np.load(cache_fn)
    else:
        file_names = [fn for fn in os.listdir(set_dirname) if fn.endswith('.wav')]
        full_sequences = []
        for fn in tqdm(file_names):
            sequence = process_wav(desired_sample_rate, set_dirname, fn)
            full_sequences.append(sequence)
        np.save(cache_fn, full_sequences)

    return full_sequences


def process_wav(desired_sample_rate, dirname, fn):
    channels = scipy.io.wavfile.read(os.path.join(dirname, fn))
    file_sample_rate, raw_audio = channels
    mono_audio = ensure_mono(raw_audio)
    mono_audio = ensure_sample_rate(desired_sample_rate, file_sample_rate, mono_audio)
    mono_audio = ulaw(mono_audio)
    return mono_audio


def ulaw(x, u=255):
    try:
        max_value = np.iinfo(x.dtype).max
    except:
        max_value = np.finfo(x.dtype).max
    uint8_max_value = np.iinfo('uint8').max
    x = x.astype('float64', casting='safe')
    x /= max_value
    x = np.sign(x) * (np.log(1 + u * np.abs(x)) / np.log(1 + u))
    x += 1.
    x /= 2.
    x *= uint8_max_value
    x = x.astype('uint8')
    return x


def ensure_sample_rate(desired_sample_rate, file_sample_rate, mono_audio):
    if file_sample_rate != desired_sample_rate:
        mono_audio = scipy.signal.resample_poly(mono_audio, desired_sample_rate, file_sample_rate).astype('uint8')
        # mono_audio = scipy.signal.resample(mono_audio, int((len(mono_audio)*desired_sample_rate) / file_sample_rate))
        # TODO: listen to output
    return mono_audio


def ensure_mono(raw_audio):
    """
    Just use first channel.
    """
    if raw_audio.shape[1] > 1:
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
                      batch_size=batch_size, fragment_stride=fragment_stride, nb_output_bins=nb_output_bins)
    print g['train'].next()[0].shape
    print g['train'].next()[1].shape
    # print len(list(g['train'])), n['train']

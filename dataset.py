"""
"""
from __future__ import division

import math
import os
import warnings

import numpy as np
import scipy.io.wavfile
import scipy.signal
from picklable_itertools import cycle
from picklable_itertools.extras import partition_all
from tqdm import tqdm


# TODO: make SACRED ingredient.
def one_hot(x):
    return np.eye(256, dtype='uint8')[x.astype('uint8')]


def fragment_indices(full_sequences, fragment_length, batch_size, fragment_stride, nb_output_bins):
    for seq_i, sequence in enumerate(full_sequences):
        # range_values = np.linspace(np.iinfo(sequence.dtype).min, np.iinfo(sequence.dtype).max, nb_output_bins)
        # digitized = np.digitize(sequence, range_values).astype('uint8')
        for i in xrange(0, sequence.shape[0] - fragment_length, fragment_stride):
            yield seq_i, i


def select_generator(set_name, random_train_batches, full_sequences, fragment_length, batch_size, fragment_stride,
                     nb_output_bins, randomize_batch_order, _rnd):
    if random_train_batches and set_name == 'train':
        bg = random_batch_generator
    else:
        bg = batch_generator
    return bg(full_sequences, fragment_length, batch_size, fragment_stride, nb_output_bins, randomize_batch_order, _rnd)


def batch_generator(full_sequences, fragment_length, batch_size, fragment_stride, nb_output_bins, randomize_batch_order, _rnd):
    indices = list(fragment_indices(full_sequences, fragment_length, batch_size, fragment_stride, nb_output_bins))
    if randomize_batch_order:
        _rnd.shuffle(indices)

    batches = cycle(partition_all(batch_size, indices))
    for batch in batches:
        if len(batch) < batch_size:
            continue
        yield np.array(
            [one_hot(full_sequences[e[0]][e[1]:e[1] + fragment_length]) for e in batch], dtype='uint8'), np.array(
            [one_hot(full_sequences[e[0]][e[1] + 1:e[1] + fragment_length + 1]) for e in batch], dtype='uint8')


def random_batch_generator(full_sequences, fragment_length, batch_size, fragment_stride, nb_output_bins,
                           randomize_batch_order, _rnd):
    lengths = [x.shape[0] for x in full_sequences]
    nb_sequences = len(full_sequences)
    while True:
        sequence_indices = _rnd.randint(0, nb_sequences, batch_size)
        batch_inputs = []
        batch_outputs = []
        for i, seq_i in enumerate(sequence_indices):
            l = lengths[seq_i]
            offset = np.squeeze(_rnd.randint(0, l - fragment_length, 1))
            batch_inputs.append(full_sequences[seq_i][offset:offset + fragment_length])
            batch_outputs.append(full_sequences[seq_i][offset + 1:offset + fragment_length + 1])
        yield one_hot(np.array(batch_inputs, dtype='uint8')), one_hot(np.array(batch_outputs, dtype='uint8'))


def generators(dirname, desired_sample_rate, fragment_length, batch_size, fragment_stride, nb_output_bins,
               learn_all_outputs, use_ulaw, randomize_batch_order, _rnd, random_train_batches):
    fragment_generators = {}
    nb_examples = {}
    for set_name in ['train', 'test']:
        set_dirname = os.path.join(dirname, set_name)
        full_sequences = load_set(desired_sample_rate, set_dirname, use_ulaw)
        fragment_generators[set_name] = select_generator(set_name, random_train_batches, full_sequences,
                                                         fragment_length,
                                                         batch_size, fragment_stride, nb_output_bins,
                                                         randomize_batch_order, _rnd)
        nb_examples[set_name] = int(sum(
            [len(xrange(0, x.shape[0] - fragment_length, fragment_stride)) for x in
             full_sequences]) / batch_size) * batch_size

    return fragment_generators, nb_examples


def generators_vctk(dirname, desired_sample_rate, fragment_length, batch_size, fragment_stride, nb_output_bins,
                    learn_all_outputs, use_ulaw, test_factor, randomize_batch_order, _rnd, random_train_batches):
    fragment_generators = {}
    nb_examples = {}
    speaker_dirs = os.listdir(dirname)
    train_full_sequences = []
    test_full_sequences = []
    for speaker_dir in speaker_dirs:
        full_sequences = load_set(desired_sample_rate, os.path.join(dirname, speaker_dir), use_ulaw)
        nb_examples_train = int(math.ceil(len(full_sequences) * (1 - test_factor)))
        train_full_sequences.extend(full_sequences[0:nb_examples_train])
        test_full_sequences.extend(full_sequences[nb_examples_train:])

    for set_name, set_sequences in zip(['train', 'test'], [train_full_sequences, test_full_sequences]):
        fragment_generators[set_name] = select_generator(set_name, random_train_batches, full_sequences,
                                                         fragment_length,
                                                         batch_size, fragment_stride, nb_output_bins,
                                                         randomize_batch_order, _rnd)
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
    return mono_audio


def ensure_mono(raw_audio):
    """
    Just use first channel.
    """
    if raw_audio.ndim == 2:
        raw_audio = raw_audio[:, 0]
    return raw_audio


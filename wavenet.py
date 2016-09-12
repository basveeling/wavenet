import json
import os
import wave

import numpy as np
from tqdm import tqdm
import keras
from keras import objectives
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras.engine import Input
from keras.engine import Model
import keras.backend as K
from keras.optimizers import Adam, SGD

import chopin22
from sacred import Experiment
from keras import layers

ex = Experiment('wavenet')


@ex.config
def config():
    data_dir = 'data'
    nb_epoch = 1000
    early_stopping_patience = nb_epoch / 10
    fragment_length = 2 ** 10
    desired_sample_rate = 4410
    batch_size = 32
    nb_output_bins = 256
    nb_filters = 32
    dilation_depth = 9  #
    nb_stacks = 1
    fragment_stride = 2 ** 11 - 3
    use_skip_connections = True
    loss_weights_mode = 'temporal_decay'
    optimizer = {
        'optimizer': 'sgd',
        'lr': 0.01,
        'momentum': 0.9,
        'decay': 0.,
        'nesterov': True
    }
    keras_verbose = 1
    debug = False


@ex.config
def predict_config():
    predict_seconds = 1
    sample_argmax = False
    sample_temperature = 0.1


@ex.named_config
def batch_run():
    keras_verbose = 2


@ex.capture()
def build_model(fragment_length, nb_filters, nb_output_bins, dilation_depth, nb_stacks, use_skip_connections):
    input = Input(shape=(fragment_length, nb_output_bins), name='input_part')
    out = input
    skip_connections = []
    out = layers.AtrousConvolution1D(nb_filters, 2, atrous_rate=1, border_mode='valid', causal=True,
                                     name='initial_causal_conv')(out)
    for s in xrange(nb_stacks):
        for i in xrange(0, dilation_depth):
            original_out = out

            # TODO: initalization, regularization?
            tanh_out = layers.AtrousConvolution1D(nb_filters, 2, atrous_rate=2 ** i, border_mode='valid', causal=True,
                                                  name='dilated_conv_%d_tanh_s%d' % (2 ** i, s))(out)
            tanh_out = layers.Activation('tanh')(tanh_out)
            sigm_out = layers.AtrousConvolution1D(nb_filters, 2, atrous_rate=2 ** i, border_mode='valid', causal=True,
                                                  name='dilated_conv_%d_sigm_s%d' % (2 ** i, s))(out)
            sigm_out = layers.Activation('sigmoid')(sigm_out)
            out = layers.Merge(mode='mul', name='gated_activation_%d_s%d' % (i, s))([tanh_out, sigm_out])
            out = layers.Convolution1D(nb_filters, 1, border_mode='same')(out)
            # TODO: how to do parameterized skip connection?
            skip_out = layers.ScalarSkipConnection()(out)
            skip_connections.append(skip_out)
            out = layers.Merge(mode='sum')([original_out, out])
    # TODO: what does 'parameterized skip connections' imply in this context?
    if use_skip_connections:
        out = layers.Merge(mode='sum')(skip_connections)
    out = layers.Activation('relu')(out)
    out = layers.Convolution1D(nb_output_bins, 1, border_mode='same')(out)
    out = layers.Activation('relu')(out)
    out = layers.Convolution1D(nb_output_bins, 1, border_mode='same')(out)
    # out = layers.Lambda(lambda x: x[:, -1, :], output_shape=(out._keras_shape[-1],))(
    #     out)  # Based on gif in deepmind blog: take last output?
    out = layers.Activation('softmax', name="output_softmax")(out)
    model = Model(input, out)
    return model


@ex.capture(prefix='optimizer')
def make_optimizer(optimizer, lr, momentum, decay, nesterov):
    if optimizer == 'sgd':
        optimizer = SGD(lr, momentum, decay, nesterov)
    else:
        raise ValueError('Invalid config for optimizer.optimizer: ' + optimizer)
    return optimizer


def run_dir_name(_seed):
    run_dir = os.path.join('models', 'run_%d' % _seed)
    return run_dir


@ex.command
def predict(desired_sample_rate, fragment_length, _log, seed, _seed, _config, predict_seconds, data_dir, batch_size,
            fragment_stride, nb_output_bins, **kwargs):
    run_dir = run_dir_name(seed)
    checkpoint_dir = os.path.join(run_dir, 'checkpoints')
    last_checkpoint = sorted(os.listdir(checkpoint_dir))[-1]
    sample_dir = os.path.join(run_dir, 'samples')
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)

    sample_file = wave.open(os.path.join(sample_dir, '%d_%d.wav' % (seed, _seed)), mode='w')
    sample_file.setnchannels(1)
    sample_file.setframerate(desired_sample_rate)
    sample_file.setsampwidth(1)

    model = build_model(fragment_length)
    model.load_weights(os.path.join(checkpoint_dir, last_checkpoint))
    data_generators, _ = chopin22.generators(data_dir, desired_sample_rate, fragment_length, batch_size,
                                             fragment_stride, nb_output_bins)
    outputs = list(data_generators['test'].next()[0][batch_size - 1])
    # outputs = [128] * fragment_length
    sample_file.writeframes(bytearray(np.argmax(outputs, axis=-1)))
    for i in tqdm(xrange(int(desired_sample_rate * predict_seconds))):
        prediction_seed = np.array(outputs[i:i + fragment_length])
        output = model.predict(np.atleast_2d(prediction_seed))
        # TODO: temperature?
        # TODO: fix new dim
        output_dist = output[0][-1]
        output_val = draw_sample(output_dist, _seed)
        outputs.append(output_val)
        print output_val,
        sample_file.writeframes(bytearray([np.arg_max(output_val)]))

    sample_file.close()

    _log.info("Done!")


@ex.capture
def draw_sample(output_dist, sample_temperature, sample_argmax, _rnd):
    if not sample_argmax:
        # output_dist /= sample_temperature
        # output_dist = np.exp(output_dist)
        output_dist /= np.sum(output_dist + 1e-7)
        output_dist = _rnd.multinomial(1, output_dist)
    return output_dist


@ex.automain
def main(data_dir, nb_epoch, early_stopping_patience, desired_sample_rate, fragment_length, batch_size, fragment_stride,
         nb_output_bins, loss_weights_mode, keras_verbose, _log, seed, _config, debug):

    if not debug:
        _log.info('Config: ')
        _log.info(_config)

        run_dir = run_dir_name(seed)
        if os.path.exists(run_dir):
            raise EnvironmentError('Run with seed %d already exists' % seed)
        os.mkdir(run_dir)
        checkpoint_dir = os.path.join(run_dir, 'checkpoints')
        json.dump(_config, open(os.path.join(run_dir, 'config.json'), 'w'))

    _log.info('Running with seed %d' % seed)

    _log.info('Loading data...')
    data_generators, nb_examples = chopin22.generators(data_dir, desired_sample_rate, fragment_length, batch_size,
                                                       fragment_stride,
                                                       nb_output_bins)

    _log.info('Building model...')
    model = build_model(fragment_length)
    _log.info(model.summary())

    optimizer = make_optimizer()
    _log.info('Compiling Model...')

    loss = objectives.categorical_crossentropy
    if loss_weights_mode:
        loss_weight = np.ones((nb_output_bins, fragment_length))
        if loss_weights_mode == 'temporal_decay':
            loss_weight = loss_weight * np.log(np.arange(1, fragment_length + 1)).reshape((1, fragment_length))
        else:
            raise ValueError("Invalid value for loss_weights_mode")
        print loss_weight
        loss = lambda t,p: loss_weight * objectives.categorical_crossentropy(t,p)
    loss.__name__ = 'weighted_categorical_crossentropy'


    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['categorical_accuracy', 'categorical_mean_squared_error'],
                  )

    # TODO: Consider gradient weighting making last outputs more important.

    callbacks = []
    if not debug:
        callbacks.extend([
            EarlyStopping(patience=early_stopping_patience, verbose=1),
            ReduceLROnPlateau(patience=early_stopping_patience / 2, cooldown=early_stopping_patience / 4, verbose=1),
            ModelCheckpoint(os.path.join(checkpoint_dir, 'checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5'),
                            save_best_only=True),
            CSVLogger(os.path.join(run_dir, 'history.csv')),
        ])

    if not debug:
        os.mkdir(checkpoint_dir)
    _log.info('Starting Training...')
    model.fit_generator(data_generators['train'],
                        nb_examples['train'],
                        nb_epoch=nb_epoch,
                        validation_data=data_generators['test'],
                        nb_val_samples=nb_examples['test'],
                        callbacks=callbacks,
                        verbose=keras_verbose)

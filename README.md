# WaveNet implementation in Keras
Based on https://deepmind.com/blog/wavenet-generative-model-raw-audio/ and https://arxiv.org/pdf/1609.03499.pdf.


[Listen to a sample 🎶!](https://soundcloud.com/basveeling/wavenet-sample)

~~Generate your own samples:

```$ KERAS_BACKEND=theano python2 wavenet.py predict with models/run_20160920_120916/config.json predict_seconds=1```~~
EDIT: The pretrained model had to be removed from the repository as it wasn't compatible with recent changes.

## Installation:
Activate a new python2 virtualenv (recommended):
```bash
pip install virtualenv
mkdir ~/virtualenvs && cd ~/virtualenvs
virtualenv wavenet
source wavenet/bin/activate
```
Clone and install requirements.
```bash
cd ~
git clone https://github.com/basveeling/wavenet.git
cd wavenet
pip install -r requirements.txt
```

Using the tensorflow backend is not recommended at this time, see [this issue](https://github.com/basveeling/wavenet/issues/7)

## Dependencies:
- [Sacred](https://github.com/IDSIA/sacred) is used for managing training and sampling. Take a look at the [documentation](http://sacred.readthedocs.io/en/latest/) for more information.

- This implementation does not support python3 as of now.

## Sampling:
Once the first model checkpoint is created, you can start sampling.

Run:
```$ KERAS_BACKEND=theano python2 wavenet.py predict with models/<your_run_folder>/config.json predict_seconds=1```

The latest model checkpoint will be retrieved and used to sample. The sample will be streamed to `[run_folder]/samples`, you can start listening when the first sample is generated.

### Sampling options:
- `predict_seconds`: float. Number of seconds to sample.
- `sample_argmax`: `True` or `False`. Always take the argmax
- `sample_temperature`: `None` or float. Controls the sampling temperature. 1.0 for the original distribution, < 1.0 for less exploitation, > 1.0 for more exploration.
- `seed`: int: Controls the seed for the sampling procedure.
- `predict_initial_input`: string: Path to a wav file, for which the first `fragment_length` samples are used as initial input.

e.g.:
```$ KERAS_BACKEND=theano python2 wavenet.py predict with models/[run_folder]/config.json predict_seconds=1```

## Training:
```$ KERAS_BACKEND=theano python2 wavenet.py```

Or for a smaller network (less channels per layer).
```$ KERAS_BACKEND=theano python2 wavenet.py with small```

### VCTK:
In order to use the VCTK dataset, first download the dataset by running `vctk/download_vctk.sh`.

Training is done with:
```$ KERAS_BACKEND=theano python2 wavenet.py with vctkdata```

For smaller network:
```$ KERAS_BACKEND=theano python2 wavenet.py with vctkdata small```

### Options:
Train with different configurations:
```$ KERAS_BACKEND=theano python2 wavenet.py with 'option=value' 'option2=value'```
Available options:
```
  batch_size = 16
  data_dir = 'data'
  data_dir_structure = 'flat'
  debug = False
  desired_sample_rate = 4410
  dilation_depth = 9
  early_stopping_patience = 20
  fragment_length = 1152
  fragment_stride = 128
  keras_verbose = 1
  learn_all_outputs = True
  nb_epoch = 1000
  nb_filters = 256
  nb_output_bins = 256
  nb_stacks = 1
  predict_initial_input = ''
  predict_seconds = 1
  predict_use_softmax_as_input = False
  random_train_batches = False
  randomize_batch_order = True
  run_dir = None
  sample_argmax = False
  sample_temperature = 1
  seed = 173213366
  test_factor = 0.1
  train_only_in_receptive_field = True
  use_bias = False
  use_skip_connections = True
  use_ulaw = True
  optimizer:
    decay = 0.0
    epsilon = None
    lr = 0.001
    momentum = 0.9
    nesterov = True
    optimizer = 'sgd'
```

## Using your own training data:
- Create a new data directory with a train and test folder in it. All wave files in these folders will be used as data.
    - Caveat: Make sure your wav files are supported by scipy.io.wavefile.read(): e.g. don't use 24bit wav and remove meta info.
- Run with: `$ python2 wavenet.py with 'data_dir=your_data_dir_name'`
- Test preprocessing results with: `$ python2 wavenet.py test_preprocess with 'data_dir=your_data_dir_name'`

## Todo:
- [ ] Local conditioning
- [ ] Global conditioning
- [x] Training on CSTR VCTK Corpus
- [x] CLI option to pick a wave file for the sample generation initial input. Done: see `predict_initial_input`.
- [x] Fully randomized training batches
- [x] Soft targets: by convolving a gaussian kernel over the one-hot targets, the network trains faster.
- [ ] Decaying soft targets: the stdev of the gaussian kernel should slowly decay.


## Uncertainties from paper:
- It's unclear if the model is trained to predict t+1 samples for every input sample, or only for the outputs for which which $t-receptive_field$ was in the input. Right now the code does the latter.
- There is no mention of weight decay, batch normalization in the paper. Perhaps this is not needed given enough data?

## Note on computational cost:
The Wavenet model is quite expensive to train and sample from. We can however trade computation cost with accuracy and fidility by lowering the sampling rate, amount of stacks and the amount of channels per layer.

For a downsized model (4000hz vs 16000 sampling rate, 16 filters v/s 256, 2 stacks vs ??):
- A Tesla K80 needs around ~4 minutes to generate one second of audio.
- A recent macbook pro needs around ~15 minutes.
Deepmind has reported that generating one second of audio with their model takes about 90 minutes.

## Disclaimer
This is a re-implementation of the model described in the WaveNet paper by Google Deepmind. This repository is not associated with Google Deepmind.

import keras.backend as K
from keras.layers.convolutional import Conv1D
from keras.utils.conv_utils import conv_output_length
import tensorflow as tf


def asymmetric_temporal_padding(x, left_pad=1, right_pad=1):
    '''Pad the middle dimension of a 3D tensor
    with "left_pad" zeros left and "right_pad" right.
    '''
    pattern = [[0, 0], [left_pad, right_pad], [0, 0]]
    return tf.pad(x, pattern)


def categorical_mean_squared_error(y_true, y_pred):
    """MSE for categorical variables."""
    return K.mean(K.square(K.argmax(y_true, axis=-1) -
                           K.argmax(y_pred, axis=-1)))


class CausalAtrousConvolution1D(Conv1D):
    def __init__(self, filters, kernel_size, init='glorot_uniform', activation=None,
                 padding='valid', strides=1, dilation_rate=1, bias_regularizer=None,
                 activity_regularizer=None, kernel_constraint=None, bias_constraint=None, use_bias=True, causal=False, **kwargs):
        super(CausalAtrousConvolution1D, self).__init__(filters,
                                                        kernel_size=kernel_size,
                                                        strides=strides,
                                                        padding=padding,
                                                        dilation_rate=dilation_rate,
                                                        activation=activation,
                                                        use_bias=use_bias,
                                                        kernel_initializer=init,
                                                        activity_regularizer=activity_regularizer,
                                                        bias_regularizer=bias_regularizer,
                                                        kernel_constraint=kernel_constraint,
                                                        bias_constraint=bias_constraint,
                                                        **kwargs)

        self.causal = causal
        if self.causal and padding != 'valid':
            raise ValueError("Causal mode dictates border_mode=valid.")

    def compute_output_shape(self, input_shape):
        input_length = input_shape[1]

        if self.causal:
            input_length += self.dilation_rate[0] * (self.kernel_size[0] - 1)

        length = conv_output_length(input_length,
                                    self.kernel_size[0],
                                    self.padding,
                                    self.strides[0],
                                    dilation=self.dilation_rate[0])

        return (input_shape[0], length, self.filters)

    def call(self, x):
        if self.causal:
            x = asymmetric_temporal_padding(x, self.dilation_rate[0] * (self.kernel_size[0] - 1), 0)
        return super(CausalAtrousConvolution1D, self).call(x)

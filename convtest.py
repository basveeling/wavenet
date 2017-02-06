import theano
import theano.tensor as T
import numpy as np
import keras.backend as K

# A test script to validate causal dilated convolutions
dilation = 2
input = T.fvector()
filters = T.fvector() # (output channels, input channels, filter rows, filter columns).
input_reshaped = T.reshape(input,(1,-1,1))
input_reshaped = K.asymmetric_temporal_padding(input_reshaped,left_pad=dilation, right_pad=0)
input_reshaped = T.reshape(input_reshaped,(1,1,-1,1))
filters_reshaped = T.reshape(filters,(1,1,-1,1))
out = T.nnet.conv2d(input_reshaped,filters_reshaped, border_mode='valid',filter_dilation=(dilation,1))
out = T.reshape(out,(1,-1,1))
out = K.asymmetric_temporal_padding(out,left_pad=dilation, right_pad=0)
out = T.reshape(out,(1,1,-1,1))
out = T.nnet.conv2d(out,filters_reshaped, border_mode='valid',filter_dilation=(dilation,1))
out = T.flatten(out)

in_input = np.arange(8,dtype='float32')
in_filters = np.array([1,1],dtype='float32')
f = theano.function([input,filters],out)
print "".join(["%3.0f" % i for i in in_input])
print "".join(["%3.0f" % i for i in f(in_input,in_filters)])

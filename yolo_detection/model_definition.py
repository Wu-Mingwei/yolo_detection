#Model Definition
#batch normalization and leaky ReLU

def batch_norm(inputs, training, data_format):
    return tf.keras.layers.BatchNormalization(
        axis = 1 if data_format == 'channels_first' else -1,
        momentum = _BATCH_NORM_DECAY, epsilon = _BATCH_NORM_EPSILON,
        scale = True
    )(inputs, training = training)

def fixed_padding(inputs, kernel_size, data_format):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg 
    if data_format == 'channels_first':
        padded_inputs = tf.pad(inputs, [[0,0], [0,0],
                                        [pad_beg, pad_end],
                                        [pad_beg, pad_end]])
    else:
        padded_inputs = tf.pad(inputs, [[0,0],
                                        [pad_beg, pad_end],
                                        [pad_beg, pad_end],
                                        [0,0]])
    return padded_inputs

def conv2d_fixed_padding(inputs, filters, kernel_size, data_format, strides = 1):
    if strides > 1:
        inputs = fixed_padding(inputs, kernel_size, data_format)
    conv = tf.keras.layers.Conv2D(
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = 'same' if strides == 1 else 'valid',
        use_bias = False,
        data_format = 'channels_last'
    )(inputs)

    conv = batch_norm(conv, training= True, data_format= data_format)
    conv = tf.keras.layers.LeakyReLU(negative_slope = _LEAKY_RELU)(conv)
    return conv 
import tensorflow as tf
from tensorflow import keras
from math import pi
import numpy as np

class PartialConvolution(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=2):
        super(PartialConvolution, self).__init__()
        self.picture_convolution = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)
        self.mask_convolution = keras.layers.Conv2D(1, kernel_size, strides=strides, padding='same', kernel_initializer = keras.initializers.Ones(), use_bias=False)
        self.mask_convolution.trainable = False
        self.bias = self.add_weight('bias', shape=[filters], trainable=True, dtype=tf.float32)
        self.kernel_size = kernel_size
        
    def call(self, inputs):
        [input_tensor, mask_tensor] = inputs
        inp_conved = self.picture_convolution(input_tensor*mask_tensor)
        mask_conved = self.mask_convolution(mask_tensor)
        inp_conved = tf.cast(inp_conved, tf.float32)
        mask_conved = tf.cast(mask_conved, tf.float32)
        out_picture = tf.math.divide_no_nan(inp_conved, mask_conved)*self.kernel_size*self.kernel_size
        out_picture = tf.cast(out_picture, tf.bfloat16)
        out_picture = out_picture + self.bias
        out_mask = tf.math.sign(mask_conved)
        return [out_picture, out_mask]

class MaskApplication(keras.layers.Layer):
    def call(self, inputs):
        [image, mask] = inputs
        return image*mask
    
class UpSampling(keras.layers.Layer):
    def __init__(self, channels):
        super(UpSampling, self).__init__()
        self.filters = np.zeros((2, 2, channels, channels))
        for i in range(channels):
            self.filters[:, :, i, i] = 1;
        self.channels = channels
        self.filters = tf.cast(tf.convert_to_tensor(self.filters), tf.float32)
    def call(self, inputs):
        shapes = tf.shape(inputs)
        return tf.nn.conv2d_transpose(inputs, self.filters, (shapes[0], shapes[1]*2, shapes[2]*2, self.channels), padding='SAME', strides=2)

def spectral(inputs):
    inputs = tf.cast(inputs, tf.complex64)
    inputs = tf.transpose(inputs, perm=[0, 3, 1, 2])
    inputs = tf.signal.fft(inputs)
    inputs = tf.transpose(inputs, perm=[0, 1, 3, 2])
    inputs = tf.signal.fft(inputs)
    inputs = tf.transpose(inputs, perm=[0, 3, 2, 1])
    inputs = tf.cast(inputs, tf.float32)/40000
    return inputs

class SpectralTransform(keras.layers.Layer):
    def __init__(self, filters):
        super(SpectralTransform, self).__init__()
        self.filters = filters
        self.conv1 = keras.layers.Conv2D(filters, 1, padding='same')
        self.conv2 = keras.layers.Conv2D(filters, 1, padding='same')
        self.conv3 = keras.layers.Conv2D(filters, 1, padding='same')
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu', dtype=tf.float32)
        self.conc = keras.layers.Concatenate()
    def call(self, inputs):
        filters = self.filters
        inputs = self.conv1(inputs)
        inputs = self.bn1(inputs)
        inputs = self.relu(inputs)
        spectred = spectral(inputs)
        spectred = self.conv2(spectred)
        spectred = self.bn2(spectred)
        spectred = self.relu(spectred)
        spectred = spectred + inputs
        spectred = self.conv3(spectred)
        return spectred 

class FFC_block(keras.layers.Layer):
    def __init__(self, filters):
        super(FFC_block, self).__init__()
        self.filters = filters
        self.conv1 = keras.layers.Conv2D(filters, 3, padding='same')
        self.conv2 = keras.layers.Conv2D(filters, 3, padding='same')
        self.conv3 = keras.layers.Conv2D(filters, 3, padding='same')
        self.conc = keras.layers.Concatenate()
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        self.relu = keras.layers.Activation('relu')
        self.fourier = SpectralTransform(filters)

    def call(self, inputs):
        filters = self.filters
        global_features = inputs[:, :, :, :filters]
        local_features = inputs[:, :, :, filters:]
        local_features = self.conv1(local_features) + self.conv2(global_features)
        global_features = self.conv3(local_features) + self.fourier(global_features)
        global_features = self.bn1(global_features)
        local_features = self.bn2(local_features)
        global_features = self.relu(global_features)
        local_features = self.relu(local_features)
        output = self.conc([global_features, local_features])
        return output

class ResNetBlock(keras.layers.Layer):
    def __init__(self, filters):
        super(ResNetBlock, self).__init__()
        self.block1 = FFC_block(filters)
        self.block2 = FFC_block(filters)
    def call(self, inputs):
        res = self.block1(inputs)
        res = self.block2(res)
        output = res + inputs
        return output
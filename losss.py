from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications import vgg16

vgg = vgg16.VGG16(include_top=False, input_shape=(128, 128, 3))
layers = vgg.layers

for layer in layers:
    layer.trainable = False

inp = keras.Input(shape=(256, 256, 3))
x = layers[1](inp)
x = layers[2](x)
x = layers[3](x)
pooling1 = keras.Model(inputs=inp, outputs=x)
inp1 = keras.Input(shape=(128, 128, 64))
x = layers[4](inp1)
x = layers[5](x)
x = layers[6](x)
pooling2 = keras.Model(inputs=inp1, outputs=x)
inp2 = keras.Input(shape=(64, 64, 128))
x = layers[7](inp2)
x = layers[8](x)
x = layers[9](x)
x = layers[10](x)
pooling3 = keras.Model(inputs=inp2, outputs=x)
def gauss_kernel(channels, kernel_size, sigma):
    ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
    return kernel
gaussian_kernel = gauss_kernel(3, 11, 1.2)
gaussian_kernel = gaussian_kernel[..., tf.newaxis]

def gaussian_blur(img, kernel_size=11, sigma=0.93):
    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1],
                                  padding='SAME', data_format='NHWC')
#@tf.function(jit_compile=True)
def hole_loss(y_true, y_pred):
    I_gt, M = y_true[0], y_true[1]
    
    return tf.norm((y_pred - I_gt), ord=1)/tf.cast(tf.size(I_gt), tf.float32)

#@tf.function(jit_compile=True)
def valid_loss(y_true, y_pred):
    I_gt, M = y_true[0], y_true[1]
    I_gt = gaussian_blur(I_gt)
    y_pred = gaussian_blur(y_pred)
    return tf.norm(y_pred - I_gt, ord=1)/tf.cast(tf.size(I_gt), tf.float32)

'''
@tf.function(jit_compile=True)
def per_los_model(I_gt, I_comp, y_pred):
    output = 0
    output += tf.norm(pooling1(I_gt) - pooling1(y_pred), ord=1)/tf.cast(tf.size(pooling1(I_gt)), tf.float32)
    output += tf.norm(pooling2(I_gt) - pooling2(y_pred), ord=1)/tf.cast(tf.size(pooling2(I_gt)), tf.float32)
    output += tf.norm(pooling3(I_gt) - pooling3(y_pred), ord=1)/tf.cast(tf.size(pooling3(I_gt)), tf.float32)
    output += tf.norm(pooling1(I_gt) - pooling1(I_comp), ord=1)/tf.cast(tf.size(pooling1(I_gt)), tf.float32)
    output += tf.norm(pooling2(I_gt) - pooling2(I_comp), ord=1)/tf.cast(tf.size(pooling2(I_gt)), tf.float32)
    output += tf.norm(pooling3(I_gt) - pooling3(I_comp), ord=1)/tf.cast(tf.size(pooling3(I_gt)), tf.float32)
    return output
'''

def ac(x):
    return tf.einsum('njim,njik->nmk', x, x)

def perceptual_style_loss(y_true, y_pred):
    I_gt, M = y_true[0], y_true[1]
    pool1_I_gt = tf.cast(pooling1(I_gt), tf.float32)
    pool2_I_gt = tf.cast(pooling2(pool1_I_gt), tf.float32)
    pool3_I_gt = tf.cast(pooling3(pool2_I_gt), tf.float32)
    pool1_pred = tf.cast(pooling1(y_pred), tf.float32)
    pool2_pred = tf.cast(pooling2(pool1_pred), tf.float32)
    pool3_pred = tf.cast(pooling3(pool2_pred), tf.float32)
    n1 = tf.cast(tf.size(pool1_I_gt), tf.float32)
    n2 = tf.cast(tf.size(pool2_I_gt), tf.float32)
    n3 = tf.cast(tf.size(pool3_I_gt), tf.float32)
    output_perceptual = 0
    output_perceptual += tf.norm(pool1_I_gt - pool1_pred, ord=1)/n1
    output_perceptual += tf.norm(pool2_I_gt - pool2_pred, ord=1)/n2
    output_perceptual += tf.norm(pool3_I_gt - pool3_pred, ord=1)/n3
    pool1_I_gt = ac(pool1_I_gt)
    pool2_I_gt = ac(pool2_I_gt)
    pool3_I_gt = ac(pool3_I_gt)
    pool1_pred = ac(pool1_pred)
    pool2_pred = ac(pool2_pred)
    pool3_pred = ac(pool3_pred)
    k1 = tf.cast(tf.shape(pool1_I_gt)[-1], tf.float32)**2
    k2 = tf.cast(tf.shape(pool2_I_gt)[-1], tf.float32)**2
    k3 = tf.cast(tf.shape(pool3_I_gt)[-1], tf.float32)**2
    output_style = 0
    output_style += tf.norm(pool1_I_gt - pool1_pred, ord=1)/(n1*k1)
    output_style += tf.norm(pool2_I_gt - pool2_pred, ord=1)/(n2*k2)
    output_style += tf.norm(pool3_I_gt - pool3_pred, ord=1)/(n3*k3)
    return (0.05*output_perceptual, 20*output_style)

'''
def perceptual_loss(y_true, y_pred):
    I_gt, M = y_true[0], y_true[1]
    I_comp = (1 - M)*y_pred + M*I_gt
    return per_los_model(I_gt, I_comp, y_pred)


@tf.function(jit_compile=True)
def st_loss_model(I_gt, I_comp, y_pred):
    output = 0
    n = tf.cast(tf.shape(pooling3(I_gt))[-1], tf.float32)
    size = tf.cast(tf.size(pooling3(I_gt)), tf.float32)
    n2 = tf.cast(tf.shape(pooling2(I_gt))[-1], tf.float32)
    n1 = tf.cast(tf.shape(pooling1(I_gt))[-1], tf.float32)
    size1 = tf.cast(tf.size(pooling1(I_gt)), tf.float32)
    size2 = tf.cast(tf.size(pooling2(I_gt)), tf.float32)
    output += tf.norm(ac(pooling1(y_pred)) - ac(pooling1(I_gt)), ord=1)/(n1**2*size1)
    output += tf.norm(ac(pooling2(y_pred)) - ac(pooling2(I_gt)), ord=1)/(size2*n2**2)
    output += tf.norm(ac(pooling3(y_pred)) - ac(pooling3(I_gt)), ord=1)/(size*n**2)
    output += tf.norm(ac(pooling1(I_comp)) - ac(pooling1(I_gt)), ord=1)/(size1*n1**2)
    output += tf.norm(ac(pooling2(I_comp)) - ac(pooling2(I_gt)), ord=1)/(size2*n2**2)
    output += tf.norm(ac(pooling3(I_comp)) - ac(pooling3(I_gt)), ord=1)/(size*n**2)
    return output

def style_loss(y_true, y_pred):
    I_gt, M = y_true[0], y_true[1]
    I_comp = (1 - M)*y_pred + M*I_gt
    return st_loss_model(I_gt, I_comp, y_pred)
'''

#@tf.function(jit_compile=True)
def total_variation_loss(y_true, y_pred):
    I_gt, M = y_true[0], y_true[1]
    I_comp = (1 - M)*y_pred + M*I_gt
    return tf.reduce_sum(tf.image.total_variation(I_comp))/tf.cast(tf.size(I_comp), tf.float32)

#@tf.function(jit_compile=True)
def PconvLoss(y_true, y_pred):
    res = perceptual_style_loss(y_true, y_pred)
    return (0*hole_loss(y_true, y_pred), 0.5*valid_loss(y_true, y_pred), res[0], res[1], total_variation_loss(y_true, y_pred))

from tkinter import *
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import PIL
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D,\
Concatenate, Conv2D, Conv2DTranspose
import tensorflow.keras.backend as K


class MaskApplication(keras.layers.Layer):
    def call(self, inputs):
        [image, mask] = inputs
        return image*mask
    

def create_model():
    input_image = keras.Input(shape=(256, 256, 1))
    input_mask = keras.Input(shape=(256, 256, 1))
    input_masked = MaskApplication()([input_image, input_mask])
    pconv1_image, pconv1_mask = PConv(64, 7)([input_masked, input_masked])
    relu1 = Activation('relu')(pconv1_image)
    pconv2_image, pconv2_mask = PConv(128, 5)([relu1, pconv1_mask])
    bn2 = BatchNormalization()(pconv2_image)
    relu2 = Activation('relu')(bn2)
    pconv3_image, pconv3_mask = PConv(256, 3)([relu2, pconv2_mask])
    bn3 = BatchNormalization()(pconv3_image)
    relu3 = Activation('relu')(bn3)
    pconv4_image, pconv4_mask = PConv(512, 3)([relu3, pconv3_mask])
    bn4 = BatchNormalization()(pconv4_image)
    relu4 = Activation('relu')(bn4)
    pconv5_image, pconv5_mask = PConv(512, 3)([relu4, pconv4_mask])
    bn5 = BatchNormalization()(pconv5_image)
    relu5 = Activation('relu')(bn5)
    pconv6_image, _ = PConv(512, 3)([relu5, pconv5_mask])
    bn6 = BatchNormalization()(pconv6_image)
    relu6 = Activation('relu')(bn6)
    decode_image_upsamp0 = Conv2DTranspose(512, 3, padding='same', strides=2)(relu6)
    bn_up0 = BatchNormalization()(decode_image_upsamp0)
    relu_up0 = LeakyReLU(0.2)(bn_up0)
    concat0 = Concatenate()([relu_up0, relu5])
    concat0 = Conv2D(512, 3, padding='same', strides=1)(concat0)
    concat0 = BatchNormalization()(concat0)
    concat0 = LeakyReLU(0.2)(concat0)
    decode_image_upsamp1 = Conv2DTranspose(512, 3, padding='same', strides=2)(concat0)
    bn_up1 = BatchNormalization()(decode_image_upsamp1)
    relu_up1 = LeakyReLU(0.2)(bn_up1)
    concat1 = Concatenate()([relu_up1, relu4])
    concat1 = Conv2D(512, 3, padding='same', strides=1)(concat1)
    concat1 = BatchNormalization()(concat1)
    concat1 = LeakyReLU(0.2)(concat1)
    decode_image_upsamp2 = Conv2DTranspose(256, 3, padding='same', strides=2)(concat1)
    bn_up2 = BatchNormalization()(decode_image_upsamp2)
    relu_up2 = LeakyReLU(0.2)(bn_up2)
    concat2 = Concatenate()([relu_up2, relu3])
    concat2 = Conv2D(256, 3, padding='same', strides=1)(concat2)
    concat2 = BatchNormalization()(concat2)
    concat2 = LeakyReLU(0.2)(concat2)
    decode_image_upsamp3 = Conv2DTranspose(128, 3, padding='same', strides=2)(concat2)
    bn_up3 = BatchNormalization()(decode_image_upsamp3)
    relu_up3 = LeakyReLU(0.2)(bn_up3)
    concat3 = Concatenate()([relu_up3, relu2])
    concat3 = Conv2D(128, 3, padding='same', strides=1)(concat3)
    concat3 = BatchNormalization()(concat3)
    concat3 = LeakyReLU(0.2)(concat3)
    decode_image_upsamp4 = Conv2DTranspose(64, 3, padding='same', strides=2)(concat3)
    bn_up4 = BatchNormalization()(decode_image_upsamp4)
    relu_up4 = LeakyReLU(0.2)(bn_up4)
    concat4 = Concatenate()([relu_up4, relu1])
    concat4 = Conv2D(64, 3, padding='same', strides=1)(concat4)
    concat4 = BatchNormalization()(concat4)
    concat4 = LeakyReLU(0.2)(concat4)
    decode_image_upsamp5 = Conv2DTranspose(32, 3, padding='same', strides=2)(concat4)
    concat5 = Concatenate()([decode_image_upsamp5, input_masked])
    output_image = Conv2D(3, 3, strides=1, dtype='float32', padding='same')(concat5)
    model = keras.Model(inputs=[input_image, input_mask], outputs=output_image)
    return model


class PConv(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=2):
        super(PConv, self).__init__()
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
        out_picture = tf.cast(out_picture, tf.float32)
        out_picture = out_picture + self.bias
        out_mask = tf.math.sign(mask_conved)
        return [out_picture, out_mask]
    

class pconvmodel(keras.Model):
    def __init__(self):
        super(pconvmodel, self).__init__()
        self.model = create_model()
        
    def call(self, x):
        x = tf.expand_dims(x, 0)
        return self.model((x, np.ones(x.shape)))
        
    def compile(self, steps_per_execution=1):
        super(pconvmodel, self).compile(steps_per_execution=steps_per_execution)

    

class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Toplevel()
        self.model = pconvmodel()
        self.model.model.load_weights('model_out1')

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)
        
        self.clear_button = Button(self.root, text='clear', command=self.clear)
        self.clear_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)

        self.c = Canvas(self.root, bg='white', width=256, height=256)
        self.c.grid(row=1, columnspan=5)
        
        self.image = Image.new("RGB", (256, 256), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        plt.imsave('input.jpg', self.image)
        output = ImageTk.PhotoImage(Image.open('input.jpg'))
        self.panel = tk.Label(self.root, image = output)   
        self.panel.grid(row=1, column=6, columnspan=5)
        self.panel.image = output
        
        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def use_pen(self):
        self.activate_button(self.pen_button)


    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)
        
    def clear(self):
        self.c.delete("all")

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
            self.draw.line([self.old_x, self.old_y, event.x, event.y], paint_color, width=self.line_width)
        self.old_x = event.x
        self.old_y = event.y
    def reset(self, event):
        self.old_x, self.old_y = None, None
        im = np.array(self.image)[:, :, 0] < 0.51
        shape = im.shape
        im = tf.convert_to_tensor(im)
        im = tf.reshape(im, (1, shape[0], shape[1], 1))
        im = tf.cast(im, tf.float32)
        im = tf.image.resize(im, (256, 256), method='nearest')
        
        output = self.model.model((im, im))
        output = np.array(tf.reshape(output, (256, 256 ,3)))
        plt.imsave('output.jpg', np.clip(output, 0, 1))
        output = ImageTk.PhotoImage(Image.open('output.jpg'))
        self.panel.configure(image=output)
        self.panel.image = output
        

Paint()

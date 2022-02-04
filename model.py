import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Concatenate, Conv2D, Conv2DTranspose
from partial_conv import PartialConvolution as PConv
from partial_conv import MaskApplication
from partial_conv import UpSampling
from losss import PconvLoss, hole_loss, valid_loss
from tensorflow.keras.losses import BinaryCrossentropy as gan_loss

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

def create_discriminator():
    input_image = keras.Input(shape=(256, 256, 3))
    input_mask = keras.Input(shape=(256, 256, 1))
    input_concated = Concatenate()([input_image, input_mask])
    conv1 = Conv2D(16, 7, strides=1, padding='same')(input_concatenated)
    bn1 = BatchNormalization()(conv1)
    lr1 = LeakyReLU(0.2)(bn1)
    conv2 = Conv2D(32, 5, strides=2, padding='same')(lr1)#128
    bn2 = BatchNormalization()(conv2)
    lr2 = LeakyReLU(0.2)(bn2)
    conv3 = Conv2D(64, 3, strides=2, padding='same')(lr2)#64
    bn3 = BatchNormalization()(conv3)
    lr3 = LeakyReLU(0.2)(bn3)
    conv4 = Conv2D(128, 3, strides=2, padding='same')(lr3)#32
    bn4 = BatchNormalization()(conv4)
    lr4 = LeakyReLU(0.2)(bn4)
    conv5 = Conv2D(256, 3, strides=2, padding='same')(lr4)#16
    bn5 = BatchNormalization()(conv5)
    lr5 = LeakyReLU(0.2)(bn5)
    conv6 = Conv2D(512, 3, strides=2, padding='same')(lr5)#8
    bn6 = BatchNormalization()(conv6)
    lr6 = LeakyReLU(0.2)(bn6)
    conv7 = Conv2D(512, 3, strides=2, padding='same')(lr6)#4
    bn7 = BatchNormalization()(conv7)
    lr7 = LeakyReLU(0.2)(bn7)
    linear_layer = tf.keras.layers.Flatten()(lr7)
    pre_last = tf.keras.layers.Dense(1024)(linear_layer)
    bn8 = BatchNormalization()(pre_last)
    lr8 = LeakyReLU(0.2)(bn8)
    output = tf.keras.layers.Dense(1, activation='softmax')(lr8)
    model = keras.Model(inputs=[input_image, input_mask], outputs=output)

class pconvmodel(keras.Model):
    def __init__(self):
        super(pconvmodel, self).__init__()
        self.model = create_model()
        
    def compile(self, optimizer, steps_per_execution=1):
        super(pconvmodel, self).compile(steps_per_execution=steps_per_execution)
        self.optimizer = optimizer
        
    def train_step(self, batch):
        images = batch[0][0]
        edges = batch[0][1]
        masks = batch[1]
        with tf.GradientTape() as tape:
            images_predicted = self.model((edges, masks))
            hole_loss, valid_loss, perceptual, style, tv_loss = PconvLoss((images, masks), images_predicted)
            loss = hole_loss + valid_loss + perceptual + style + 0.1*tv_loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return {'Total loss':loss, 'Hole':hole_loss, 'Valid':valid_loss, 'Perceptual': perceptual, 'Style':style, 'Total variation':0.1*tv_loss}

    def test_step(self, batch):
        images = batch[0]
        masks = batch[1]
        images_predicted = self.model((images, masks))
        loss = PconvLoss((images, masks), images_predicted)
        return {'Loss':loss}
    
class joint_train(pconvmodel):
    def train_step(self, batch):
        images = batch[0]
        masks = batch[1]
        batch_size = tf.shape(images)[0]
        batch_size = tf.cast(batch_size*0.5, tf.int64)
        images_for_generator = images[0:batch_size]
        images_for_discriminator = images[batch_size:-1]
        masks = masks[0:batch_size]
        with tf.GradientTape() as tape:
            images_predicted = self.model((images_for_generator, masks))
            real_predicts = self.discriminator((images_for_discriminator, masks))
            fake_predicts = self.discriminator((images_predicted, masks))
            real_loss = self.disc_loss(tf.ones(shape=(batch_size, 1)), real_predicts)
            fake_loss = self.disc_loss(tf.zeros(shape=(batch_size, 1)), fake_predicts)
            disc_loss = real_loss + fake_loss
            gen_loss = self.gen_loss(tf.ones(shape=(batch_size, 
                                                    1)), fake_predicts) + 60*valid_loss((images_for_generator, 
                                                                                      masks), images_predicted) + hole_loss((images_for_generator, 
                                                                                                                             masks), images_predicted)
            loss = self.g_w*gen_loss + self.d_w*disc_loss
        grads = tape.gradient(loss*d_w, self.discriminator.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        grads = tape.gradient(loss*g_w, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        t = self.g_w
        self.g_w = self.d_w
        self.d_w = t
        return {"gen_loss": gen_loss, "disc_loss":disc_loss}
    
class pretrain_discriminator(pconvmodel):
    def train_step(self, batch):
        images = batch[0]
        masks = batch[1]
        batch_size = tf.shape(images)[0]
        batch_size = tf.cast(batch_size*0.5, tf.int64)
        images_for_generator = images[0:batch_size]
        images_for_discriminator = images[batch_size:-1]
        masks = masks[0:batch_size]
        with tf.GradientTape() as tape:
            real_loss = self.gen_loss(tf.ones(shape=(batch_size, 1)), self.discriminator((images_for_discriminator, masks)))
            fake_loss = self.gen_loss(tf.zeros(shape=(batch_size, 1)), self.discriminator((self.model((images_for_generator, masks)), masks)))
            loss = real_loss + fake_loss
        grads = tape.gradient(loss, self.discriminator.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        return {"loss": loss}
        
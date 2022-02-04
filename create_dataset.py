from tensorflow import keras
import tensorflow as tf
from tensorflow.data import AUTOTUNE 
import tensorflow.keras.backend as K
import math

def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear = math.pi * shear / 180.
    
    # ROTATION MATRIX
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )
        
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    
    
    # ZOOM MATRIX
    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )
    
    # SHIFT MATRIX
    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))

def transform(image, m):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = 256

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack( [x,y,z] )

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))
    idx2 = K.cast(idx2,dtype='int32')
    idx2 = K.clip(idx2,-DIM//2 + 1,DIM//2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
    d = tf.gather_nd(image,tf.transpose(idx3))

    return tf.reshape(d,[DIM,DIM,3])

def transform_edges(image, m):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = 256

    # LIST DESTINATION PIXEL INDICES
    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )
    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )
    z = tf.ones([DIM*DIM],dtype='int32')
    idx = tf.stack( [x,y,z] )

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))
    idx2 = K.cast(idx2,dtype='int32')
    idx2 = K.clip(idx2,-DIM//2 + 1,DIM//2)

    # FIND ORIGIN PIXEL VALUES
    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )
    d = tf.gather_nd(image,tf.transpose(idx3))

    return tf.reshape(d,[DIM,DIM,1])

def decode_image(image_data, m):
    image = tf.io.parse_tensor(image_data, out_type=tf.uint8)/255
    image = tf.reshape(image, (256, 256, 3))
    image = transform(image, m)
    return image

def decode_edges(image_data, m):
    image = tf.io.parse_tensor(image_data, out_type=tf.uint8)
    image = tf.reshape(image, (256, 256, 1))
    image = transform_edges(image, m)
    return image

def decode_mask(mask_data):
    image = tf.io.parse_tensor(mask_data, out_type=tf.uint8)
    image = tf.reshape(image, (256, 256, 1))
    image = tf.cast(image, tf.float32)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    threshold = tf.random.uniform(shape=(), maxval=0.95)
    matrix = tf.random.uniform(shape=(256, 256), maxval=1)
    image = tf.reshape(image, (256, 256))
    image = tf.math.multiply(image, tf.math.sign(tf.nn.relu(matrix - threshold)))
    image = tf.reshape(image, (256, 256, 1))
    return image

def read_tfrecord_image(example):
    TFREC_FORMAT = {
            'image': tf.io.FixedLenFeature([], tf.string),
            "egdes": tf.io.FixedLenFeature([], tf.string),
            }
    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    rot = 20. * tf.random.normal([1],dtype='float32')
    shr = 5. * tf.random.normal([1],dtype='float32')
    h_zoom = 1. + tf.random.normal([1],dtype='float32')/10.
    w_zoom = 1. + tf.random.normal([1],dtype='float32')/10.
    h_shift = 16. * tf.random.normal([1],dtype='float32')
    w_shift = 16. * tf.random.normal([1],dtype='float32')
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift)
    image = decode_image(example['image'], m)
    edges = decode_edges(example['egdes'], m)
    return (image, edges)


def get_dataset(mask_filenames, image_filenames, batch_size=64):
    masks_data = tf.data.TFRecordDataset(mask_filenames, num_parallel_reads=AUTOTUNE).shuffle(5000, reshuffle_each_iteration=True)
    image_data = tf.data.TFRecordDataset(image_filenames, num_parallel_reads=AUTOTUNE).shuffle(5000, reshuffle_each_iteration=True)
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    masks_data = masks_data.with_options(ignore_order)
    image_data = image_data.with_options(ignore_order)
    masks_data = masks_data.map(decode_mask, num_parallel_calls=AUTOTUNE).repeat()
    image_data = image_data.repeat().map(read_tfrecord_image, num_parallel_calls=AUTOTUNE)
    output_dataset = tf.data.TFRecordDataset.zip((image_data, masks_data)).batch(batch_size=batch_size, drop_remainder=True)
    return output_dataset.prefetch(AUTOTUNE)
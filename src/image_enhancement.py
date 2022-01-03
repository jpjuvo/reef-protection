import tensorflow as tf
import numpy as np
import cv2
import os

# Copied and modified from here:
# https://www.kaggle.com/c/tensorflow-great-barrier-reef/discussion/290584
@tf.function(input_signature=(tf.TensorSpec(shape=[720,1280,3], dtype=tf.uint8),))
def equalize(image):
    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = tf.cast(im[..., c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(
            tf.equal(step, 0), lambda: im,
            lambda: tf.gather(build_lut(histo, step), im))
        return tf.cast(result, tf.uint8)
    
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], -1)
    return tf.cast(image, tf.float32)

def tf_clahe(image):
    """ Converts rgb to rgb-clahe (input and output in uint8 rgb) """
    if image.shape[0] != 720 or image.shape[1] != 1280:
        image = cv2.resize(image, (1280,720)) 
    tf_image = tf.convert_to_tensor(image, dtype=tf.uint8)
    image = equalize(tf_image).numpy().astype(np.uint8)
    del tf_image
    return image
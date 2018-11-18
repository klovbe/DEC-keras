from keras.layers import Dense, Input
from keras.models import Model
import keras.backend as K
import tensorflow as tf
import numpy as np

x = Input(shape=(2,))
mask = K.equal(x,0)
# mask = K.cast(mask,dtype='float32')
# y = x*mask
y = tf.boolean_mask(x, mask, axis=)
# y = tf.reshape(y, [-1,1])

with tf.Session() as sess:
    x = sess.run([x,y], feed_dict={x:np.array([0,2,3,0,0,4,5,6]).reshape(4,2)})
    print(x)

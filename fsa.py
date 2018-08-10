import tensorflow as tf
import numpy as np

y = [
    [
        [3, 4, 2],
        [3, 4, 0]
    ],
    [
        [3, 4, 2],
        [3, 4, 0]
    ],
]

f = [2, 2, 2]
# 2 * 2 * 3


m = tf.Variable(np.array(y), dtype=tf.float32)
ff = tf.Variable(np.array(f), dtype=tf.float32)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run(tf.multiply(m, ff)))
    x = sess.run(tf.reduce_sum(tf.multiply(m, ff), axis=2, keepdims=True))
    print(tf.nn.softmax(x, dim=1))

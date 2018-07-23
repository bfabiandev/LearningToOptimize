import tensorflow as tf


def preprocess_grads(gradients):
    # if it throws error, potentially add epsilon to avoid log(0)
    p = 10.0
    eps = 1e-6
    indicator = tf.to_float(tf.abs(gradients) > tf.exp(-p))
    log = (tf.log(tf.abs(gradients) + eps) / p) * indicator - (1 - indicator)
    sign = tf.sign(gradients) * indicator - tf.exp(p) * \
        gradients * (1 - indicator)

    return tf.concat([log, sign], 1)

import tensorflow as tf

import time
import numpy as np


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(seq_length, d_model):
    """
     The formula cossin for positional encoding
    :param seq_length:
    :param d_model:
    :return:
    """
    angle_rads = get_angles(np.arange(seq_length)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)

    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def positional_embedding(
        seq_length,
        d_model,
        position_embedding_name="position_embeddings",
        max_position_embeddings=512,
        initial_stddev=0.02):
    full_position_embeddings = tf.Variable(
        initial_value=tf.random.truncated_normal(shape=[max_position_embeddings, d_model], stddev=initial_stddev),
        name=position_embedding_name)
    # Since the position embedding table is a learned variable, we create it
    # using a (long) sequence length `max_position_embeddings`. The actual
    # sequence length might be shorter than this, for faster training of
    # tasks that do not have long sequences.
    #
    # So `full_position_embeddings` is effectively an embedding table
    # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
    # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
    # perform a slice.
    position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                   [seq_length, -1])

    pos_encoding = position_embeddings[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

import tensorflow as tf


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def test():
    seq = tf.constant([[0, 1, 2], [1, 2, 3]])
    dec_target_padding_mask = create_padding_mask(seq)
    look_ahead_mask = create_look_ahead_mask(seq.shape[1])

    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    print(combined_mask)


test()

import tensorflow as tf
from absl import flags
from absl import app
from absl import logging

from tokenization import FullTokenizer
from tokenization_en import load_subword_vocab
from transformer import Transformer, FileConfig

FLAGS = flags.FLAGS

MODEL_DIR = "/Users/livingmagic/Documents/deeplearning/models/bert-nmt/zh-en_bert-tf2_L6-D256/"

flags.DEFINE_string("bert_config_file", MODEL_DIR + "bert_config.json", "The bert config file")
flags.DEFINE_string("bert_vocab_file", MODEL_DIR + "vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("init_checkpoint", MODEL_DIR + "bert_nmt_ckpt", "")
flags.DEFINE_string("config_file", MODEL_DIR + "config.json", "The transformer config file except bert")
flags.DEFINE_string("vocab_file", MODEL_DIR + "vocab_en", "The english vocabulary file")
flags.DEFINE_integer("max_seq_length", 128, "Max length to sequence length")
flags.DEFINE_string("inp_sentence", None, "")


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


def create_look_ahead_mask(size):
    """
    The look-ahead mask is used to mask the future tokens in a sequence.
    In other words, the mask indicates which entries should not be used.
    """
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


def create_masks(inp, tar):
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return combined_mask, dec_padding_mask


def encode_zh(tokenizer_zh, zh):
    tokens_zh = tokenizer_zh.tokenize(zh)
    lang1 = tokenizer_zh.convert_tokens_to_ids(['[CLS]'] + tokens_zh + ['[SEP]'])

    return lang1


def evaluate(transformer,
             tokenizer_zh,
             tokenizer_en,
             inp_sentence,
             max_seq_length):
    # normalize input sentence
    inp_sentence = encode_zh(tokenizer_zh, inp_sentence)
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the
    # english start token.
    decoder_input = [tokenizer_en.vocab_size]
    output = tf.expand_dims(decoder_input, 0)

    for i in range(max_seq_length):
        combined_mask, dec_padding_mask = create_masks(
            encoder_input, output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last word from the seq_len dimension
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        # return the result if the predicted_id is equal to the end token
        if tf.equal(predicted_id, tokenizer_en.vocab_size + 1):
            return tf.squeeze(output, axis=0), attention_weights

        # concatentate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def main(_):
    tokenizer_zh = FullTokenizer(
        vocab_file=FLAGS.bert_vocab_file, do_lower_case=True)

    tokenizer_en = load_subword_vocab(FLAGS.vocab_file)
    target_vocab_size = tokenizer_en.vocab_size + 2

    config = FileConfig(FLAGS.config_file)
    transformer = Transformer(config=config,
                              target_vocab_size=target_vocab_size,
                              bert_config_file=FLAGS.bert_config_file)

    inp = tf.random.uniform((1, FLAGS.max_seq_length))
    tar_inp = tf.random.uniform((1, FLAGS.max_seq_length))
    fn_out, _ = transformer(inp, tar_inp,
                            True,
                            look_ahead_mask=None,
                            dec_padding_mask=None)

    transformer.load_weights(FLAGS.init_checkpoint)

    print(transformer.encoder.weights[0])

    result, _ = evaluate(transformer,
                         tokenizer_zh,
                         tokenizer_en,
                         FLAGS.inp_sentence,
                         FLAGS.max_seq_length)

    predicted_sentence = tokenizer_en.decode([i for i in result
                                              if i < tokenizer_en.vocab_size])

    print('Input: {}'.format(FLAGS.inp_sentence))
    print('Predicted translation: {}'.format(predicted_sentence))


if __name__ == "__main__":
    flags.mark_flag_as_required("inp_sentence")
    app.run(main)

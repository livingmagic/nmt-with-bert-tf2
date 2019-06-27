from transformer import *
import numpy as np
import os


class TransformerTest(tf.test.TestCase):
    def test_transform(self):
        MODEL_DIR = "/Users/livingmagic/Documents/deeplearning/models/bert/chinese_L-12_H-768_A-12"
        bert_config_file = os.path.join(MODEL_DIR, "bert_config.json")
        bert_ckpt_file = os.path.join(MODEL_DIR, "bert_model.ckpt")

        config = Config(num_layers=4, d_model=128, dff=512, num_heads=8)

        transformer = Transformer(config=config,
                                  target_vocab_size=8173,
                                  bert_config_file=bert_config_file)

        inp = tf.random.uniform((32, 128))
        tar_inp = tf.random.uniform((32, 128))
        fn_out, _ = transformer(inp, tar_inp,
                                True,
                                look_ahead_mask=None,
                                dec_padding_mask=None)
        print(tar_inp.shape)
        print(fn_out.shape)  # (batch_size, tar_seq_len) (batch_size, tar_seq_len, target_vocab_size)

        w11 = tf.reduce_sum(transformer.encoder.weights[0]).numpy()
        w12 = tf.reduce_sum(transformer.encoder.weights[1]).numpy()
        # init bert pre-trained weights
        transformer.restore_encoder(bert_ckpt_file)
        w21 = tf.reduce_sum(transformer.encoder.weights[0]).numpy()
        w22 = tf.reduce_sum(transformer.encoder.weights[1]).numpy()
        self.assertNotEqual(w11, w21)
        self.assertNotEqual(w12, w22)

    def test_encoder(self):
        MODEL_DIR = "/Users/livingmagic/Documents/deeplearning/models/bert/chinese_L-12_H-768_A-12"
        bert_config_file = os.path.join(MODEL_DIR, "bert_config.json")

        bert_encoder = build_encoder(config_file=bert_config_file)
        bert_encoder.trainable = False
        inp = tf.random.uniform((32, 128))
        bert_encoder(inp, training=False)

        weight_names = []
        for weight in bert_encoder.weights:
            weight_names.append(weight.name)
        with open('2.txt', 'w') as f:
            f.write(str(weight_names))


if __name__ == "__main__":
    tf.test.main()

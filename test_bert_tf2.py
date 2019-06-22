import os
import tensorflow as tf
import numpy as np
from bert.embeddings import BertEmbeddingsLayer
from tensorflow import keras
from bert import BertModelLayer
from bert.loader import StockBertConfig, load_stock_weights
import logging


def test1():
    l_bert = BertModelLayer(
        vocab_size=16000,  # embedding params
        use_token_type=True,
        use_position_embeddings=True,
        token_type_vocab_size=2,

        num_layers=12,  # transformer encoder params
        hidden_size=768,
        hidden_dropout=0.1,
        intermediate_size=4 * 768,
        intermediate_activation="gelu",

        name="bert"  # any other Keras layer params
    )

    print(l_bert.params)


def test2():
    model_dir = "/Users/livingmagic/Documents/deeplearning/models/bert/chinese_L-12_H-768_A-12"

    bert_config_file = os.path.join(model_dir, "bert_config.json")
    bert_ckpt_file = os.path.join(model_dir, "bert_model.ckpt")

    with tf.io.gfile.GFile(bert_config_file, "r") as reader:
        stock_params = StockBertConfig.from_json_string(reader.read())
        bert_params = stock_params.to_bert_model_layer_params()

    l_bert = BertModelLayer.from_params(bert_params, name="bert", trainable=False)

    # # Input and output endpoints
    max_seq_len = 128
    l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32')
    output = l_bert(l_input_ids, training=False)  # [batch_size, max_seq_len, hidden_size]
    print('Output shape: {}'.format(output.get_shape()))

    bert_model = keras.Model(inputs=l_input_ids, outputs=output)
    print(bert_model.trainable_weights)

    # # loading the original pre-trained weights into the BERT layer:
    # load_stock_weights(l_bert, bert_ckpt_file)
    # print(bert_model.predict(np.arange(0, 128)[np.newaxis, :]))


def test_embeddings_layer():
    tf.get_logger().setLevel(logging.INFO)
    layer = BertEmbeddingsLayer()
    mask = layer.compute_mask(inputs=np.array([1, 2, 0, 0, 1]))
    print('mask: {}'.format(tf.cast(mask, tf.float32) * 0.1))


test_embeddings_layer()
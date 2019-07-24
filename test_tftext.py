import tensorflow as tf
import numpy as np
import tokenization

tokenizer = tokenization.FullTokenizer(
        vocab_file='/Users/livingmagic/Documents/deeplearning/models/bert/chinese_L-12_H-768_A-12/vocab.txt', do_lower_case=True)

tokens = tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
print(tokens)

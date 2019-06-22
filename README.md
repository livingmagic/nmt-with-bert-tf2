# nmt-with-bert-tf2
A Transformer model to translate Chinese to English using pre-trained model BERT as encoder in Tensorflow 2.0.

## Usage

you can use the Google Colab notebook: [nmt_with_transformer.ipynb](https://colab.research.google.com/github/livingmagic/nmt-with-bert-tf2/blob/master/nmt_with_transformer.ipynb)



## Using BERT to extract fixed feature vectors (like ELMo)

The chinese BERT pre-trained model: 
https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip

```
# Sentence A and Sentence B are separated by the ||| delimiter for sentence
# pair tasks like question answering and entailment.
# For single sentence inputs, put one sentence per line and DON'T use the
# delimiter.
echo '我不是故意说的。 ||| 真心话大冒险！\n富士康科技集团发声明否认“撤离大陆”?' > /tmp/input.txt

python extract_features.py \
  --input_file=tmp/input.txt \
  --output_file=tmp/output.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --batch_size=8
```

## Resources

- [BERT](https://arxiv.org/abs/1810.04805) - BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- [google-research/bert](https://github.com/google-research/bert) - the original BERT implementation
- [kpe/bert-for-tf2](https://github.com/kpe/bert-for-tf2) - A Keras TensorFlow 2.0 implementation of BERT. 
# nmt-with-bert-tf2
A Transformer model to translate Chinese to English using pre-trained model BERT as encoder in Tensorflow 2.0.

## Usage

You can train model free in the Google Colab notebook: [nmt_with_transformer.ipynb](https://colab.research.google.com/github/livingmagic/nmt-with-bert-tf2/blob/master/nmt_with_transformer.ipynb)

After 4 epochs, the final training result is: **Epoch 4 Loss 0.5936 Accuracy 0.1355**

```reStructuredText
Epoch 4 Batch 0 Loss 0.6691 Accuracy 0.1323
Epoch 4 Batch 500 Loss 0.6059 Accuracy 0.1335
Epoch 4 Batch 1000 Loss 0.6038 Accuracy 0.1335
Epoch 4 Batch 1500 Loss 0.6016 Accuracy 0.1335
Epoch 4 Batch 2000 Loss 0.6010 Accuracy 0.1342
Epoch 4 Batch 2500 Loss 0.5993 Accuracy 0.1346
Epoch 4 Batch 3000 Loss 0.5968 Accuracy 0.1350
Epoch 4 Batch 3500 Loss 0.5955 Accuracy 0.1353
Saving checkpoint for epoch 4 at ./checkpoints/train/ckpt-4
Epoch 4 Loss 0.5936 Accuracy 0.1355
Time taken for 1 epoch: 3553.9979977607727 secs
```

You can evaluate some texts, such as:

```reStructuredText
Input: 我爱你是一件幸福的事情。
Predicted translation: I love you are a blessing.
```

```text
Input: 虽然继承了祖荫，但朴槿惠已经证明了自己是个机敏而老练的政治家——她历经20年才爬上韩国大国家党最高领导层并成为全国知名人物。
Predicted translation: While inherited her father, Park has proven that she is a brave and al-keal – a politician who has been able to topple the country’s largest party and become a national leader.
Real translation: While Park derives some of her power from her family pedigree, she has proven to be an astute and seasoned politician –&nbsp;one who climbed the Grand National Party’s leadership ladder over the last two decades to emerge as a national figure.
```

**Notice**: Must add"。"in the end of the input sentence, otherwise you may get a unexpected result.

## Using BERT to extract fixed feature vectors (like ELMo)

The chinese BERT pre-trained model used here is: [BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip). You can use the example code below to extract features using tensorflow2.

```
# Sentence A and Sentence B are separated by the ||| delimiter for sentence
# pair tasks like question answering and entailment.
# For single sentence inputs, put one sentence per line and DON'T use the
# delimiter.
echo '我不是故意说的。 ||| 真心话大冒险！\n富士康科技集团发声明否认“撤离大陆”?' > tmp/input.txt

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
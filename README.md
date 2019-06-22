# text-models
For text models using tensorflow 2.0

## Using BERT to extract fixed feature vectors (like ELMo)

The chinese BERT pre-trained model: https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip

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

## Loading zh-en translation dataset
```python
import tensorflow_datasets as tfds

data, info = tfds.load("wmt19_translate/zh-en", with_info=True)
print(info)
```

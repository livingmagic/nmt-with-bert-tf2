import tensorflow_datasets as tfds


def load_dataset():
    config = tfds.translate.wmt.WmtConfig(
        description="WMT 2019 translation task dataset.",
        version="0.0.3",
        language_pair=("zh", "en"),
        subsets={
            tfds.Split.TRAIN: ["newscommentary_v13"],
            tfds.Split.VALIDATION: ["newsdev2017"],
        }
    )

    builder = tfds.builder("wmt_translate", config=config)
    print(builder.info.splits)
    builder.download_and_prepare()
    datasets = builder.as_dataset(as_supervised=True)

    return datasets['train'], datasets['validation']


def make_subword_vocab(examples):
    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for zh, en in examples), target_vocab_size=2 ** 13)
    sample_string = 'Transformer is awesome.'
    tokenized_string = tokenizer_en.encode(sample_string)
    print('Tokenized string is {}'.format(tokenized_string))
    original_string = tokenizer_en.decode(tokenized_string)
    print('The original string: {}'.format(original_string))

    tokenizer_en.save_to_file('vocab_en')


def load_subword_vocab(vocab_file):
    return tfds.features.text.SubwordTextEncoder.load_from_file(vocab_file)


if __name__ == "__main__":
    train_examples, _ = load_dataset()
    tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        (en.numpy() for zh, en in train_examples), target_vocab_size=2 ** 13)
    tokenizer_en.save_to_file('vocab_en.txt')
    print(tokenizer_en.vocab_size + 2)

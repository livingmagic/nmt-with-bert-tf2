import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as text

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
print(builder.info)
builder.download_and_prepare()
datasets = builder.as_dataset(as_supervised=True)
train_dataset = datasets['train']
val_dataset = datasets['validation']

for zh, en in train_dataset.take(5):
    print('zh: {}'.format(zh.numpy()))
    print('en: {}'.format(en.numpy()))

# If you need NumPy arrays
# np_datasets = tfds.as_numpy(datasets)

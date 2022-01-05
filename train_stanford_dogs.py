import tensorflow as tf
import matplotlib.pyplot as plt
import math
import time
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LambdaCallback

from MobileNet import MyMobileNet

# Stanford dogs
# https://www.tensorflow.org/datasets/catalog/stanford_dogs

IMG_LEN = 224
IMG_SHAPE = (IMG_LEN,IMG_LEN,3)
N_CLASSES = 120
DATA_DIR = "C:\\StanfordDogs"


print(tf.__version__)
def preprocess(ds_row):
  
    # Image conversion int->float + resizing
    image = tf.image.convert_image_dtype(ds_row['image'], dtype=tf.float32)
    image = tf.image.resize(image, (IMG_LEN, IMG_LEN), method='nearest')
  
    # Onehot encoding labels
    label = tf.one_hot(ds_row['label'], N_CLASSES)

    return image, label

def prepare(dataset, batch_size=None):
    ds = dataset.map(preprocess, num_parallel_calls=4)
    ds = ds.shuffle(buffer_size=1000)
    if batch_size:
      ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds


if __name__ == '__main__':
    my_mobilenet = MyMobileNet()
    my_mobilenet.build((None, *IMG_SHAPE))
    my_mobilenet.summary()

    my_mobilenet_seq = tf.keras.Sequential([
        my_mobilenet,
        tf.keras.layers.Dense(N_CLASSES, activation='softmax')
    ])

    my_mobilenet_seq.build([None, *IMG_SHAPE])

    my_mobilenet_seq.compile(optimizer=tf.keras.optimizers.RMSprop(0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'top_k_categorical_accuracy']
    )
                
    
    dataset = tfds.load(name="stanford_dogs", data_dir=DATA_DIR, download=True)

    training_data = dataset['train']
    test_data = dataset['test']

    train_batches = prepare(training_data, batch_size=32)
    test_batches = prepare(test_data, batch_size=32)

    save_weights_callback = LambdaCallback(
                on_epoch_end=lambda epoch, logs: my_mobilenet.save_weights(f'mobilenet_weights_{epoch}', overwrite=True)
    )

    # my_mobilenet.load_weights("loss_log_0.json")
    # my_mobilenet.trainable = False

    history = my_mobilenet_seq.fit(train_batches,
                        epochs=30,
                        validation_data=test_batches,
                        callbacks=[save_weights_callback]
    )
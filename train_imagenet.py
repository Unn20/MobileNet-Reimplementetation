import tensorflow as tf
import matplotlib.pyplot as plt
import math
import time
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import LambdaCallback

from MobileNet import MyMobileNet

IMG_LEN = 224
IMG_SHAPE = (IMG_LEN,IMG_LEN,3)
N_CLASSES = 1000

if __name__ == '__main__':
    my_mobilenet = MyMobileNet()
    my_mobilenet.build((None, *IMG_SHAPE))
    my_mobilenet.summary()

import tensorflow as tf
import math

class MyDepthwiseSeparableConvolution(tf.keras.layers.Layer):
  def __init__(self, output_features, strides=(1, 1)):
    super(MyDepthwiseSeparableConvolution, self).__init__()
    self.depthwise_conv = tf.keras.layers.DepthwiseConv2D(kernel_size=(3, 3), strides=strides,
                                                          padding='same')

    self.pointwise_conv = tf.keras.layers.Conv2D(filters=output_features, kernel_size=(1, 1),
                                                 padding='same')

    self.bn1 = tf.keras.layers.BatchNormalization()
    self.bn2 = tf.keras.layers.BatchNormalization()

  def call(self, inputs, training=None):
    x1 = self.depthwise_conv(inputs)
    x1 = self.bn1(x1, training=training)
    x1 = tf.nn.relu(x1)
    x2 = self.pointwise_conv(x1)
    x2 = self.bn2(x2, training=training)
    return tf.nn.relu(x2)


class MyMobileNet(tf.keras.Model):
  def __init__(self, alpha=1.0, depth_multiplier=1.0):
    super(MyMobileNet, self).__init__()

    new_size = math.ceil(depth_multiplier * 224)

    self.input_layer = tf.keras.layers.InputLayer(input_shape=(new_size,new_size,3))
    self.conv1 = tf.keras.layers.Conv2D(math.ceil(alpha * 32), (3, 3), strides=(2, 2), padding='same')
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.ds_conv1 = MyDepthwiseSeparableConvolution(math.ceil(alpha * 64), (1, 1))
    self.ds_conv2 = MyDepthwiseSeparableConvolution(math.ceil(alpha * 128), (2, 2))
    self.ds_conv3 = MyDepthwiseSeparableConvolution(math.ceil(alpha * 128), (1, 1))
    self.ds_conv4 = MyDepthwiseSeparableConvolution(math.ceil(alpha * 256), (2, 2))
    self.ds_conv5 = MyDepthwiseSeparableConvolution(math.ceil(alpha * 256), (1, 1))
    self.ds_conv6 = MyDepthwiseSeparableConvolution(math.ceil(alpha * 256), (1, 1))
    self.ds_conv7 = MyDepthwiseSeparableConvolution(math.ceil(alpha * 512), (2, 2))
    self.ds_conv8 = MyDepthwiseSeparableConvolution(math.ceil(alpha * 512), (1, 1))
    self.ds_conv9 = MyDepthwiseSeparableConvolution(math.ceil(alpha * 512), (1, 1))
    self.ds_conv10 = MyDepthwiseSeparableConvolution(math.ceil(alpha * 512), (1, 1))
    self.ds_conv11 = MyDepthwiseSeparableConvolution(math.ceil(alpha * 512), (1, 1))
    self.ds_conv12 = MyDepthwiseSeparableConvolution(math.ceil(alpha * 512), (1, 1))
    self.ds_conv13 = MyDepthwiseSeparableConvolution(math.ceil(alpha * 1024), (2, 2))
    self.ds_conv14 = MyDepthwiseSeparableConvolution(math.ceil(alpha * 1024), (1, 1))

    self.avg_pooling = tf.keras.layers.GlobalAvgPool2D()
  
  def call(self, inputs, training=None, **kwargs):
    inputs = self.input_layer(inputs)
    x = self.conv1(inputs)
    x = self.bn1(x, training=training)
    x = tf.nn.relu(x)
    x = self.ds_conv1(x)
    x = self.ds_conv2(x)
    x = self.ds_conv3(x)
    x = self.ds_conv4(x)
    x = self.ds_conv5(x)
    x = self.ds_conv6(x)
    x = self.ds_conv7(x)
    x = self.ds_conv8(x)
    x = self.ds_conv9(x)
    x = self.ds_conv10(x)
    x = self.ds_conv11(x)
    x = self.ds_conv12(x)
    x = self.ds_conv13(x)
    x = self.ds_conv14(x)
    x = self.avg_pooling(x)
    return x
# import tensorflow as tf
import tensorflow as tf
import ops
import utils

class Generator:
  def __init__(self, name, is_training, ngf=64, norm='instance', image_size1=128, image_size2=128):
    self.name = name
    self.reuse = False
    self.ngf = ngf
    self.norm = norm
    self.is_training = is_training
    self.image_size1 = image_size1
    self.image_size2 = image_size2

  def __call__(self, input):
    """
    Args:
      input: batch_size x width x height x 3
    Returns:
      output: same size as input
    """
    with tf.variable_scope(self.name):
      
      output= ops.candyconv(input,reuse=self.reuse,is_training=self.is_training,name='genconv');
    # set reuse=True for next call
    self.reuse = True
    self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    return output

  def sample(self, input):
    image = utils.batch_convert2int(self.__call__(input))
    image = tf.image.encode_jpeg(tf.squeeze(image, [0]))
    return image


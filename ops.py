import tensorflow as tf
## Layers: follow the naming convention used in the original paper
### Generator layers

def candyconv(input, reuse=False, is_training=True, name='genconv'):
  output1=[]
  input1=input;
  for i in range(0,6):
    if i == 0:
      output1.append(conv3_3(input,'conv3x3_{}'.format(i), reuse=reuse,is_training=is_training, activation='Prelu',norm='none'))
    else:
      output1.append(conv3_3(input,'conv3x3_{}'.format(i), reuse=reuse,is_training=is_training, activation='Prelu'))
    input = output1[i]
  output2=[]
  for i in range(0,6):
    if i == 1:
      output2.append(conv3_3(input,'deconv3x3_{}'.format(i), reuse=reuse,is_training=is_training, activation='relu',prev_input=output1[3]))
    elif i == 3:
      output2.append(conv3_3(input,'deconv3x3_{}'.format(i), reuse=reuse,is_training=is_training, activation='relu',prev_input=output1[1]))
    elif i == 5:
      output2.append(conv3_3(input,'deconv3x3_{}'.format(i), reuse=reuse,is_training=is_training, activation='relu',prev_input=input1, k=3))
      # output2.append(conv3_3(input,'deconv3x3_{}'.format(i), reuse=reuse,is_training=is_training, activation='relu'))
    else:
      output2.append(conv3_3(input,'deconv3x3_{}'.format(i), reuse=reuse,is_training=is_training, activation='relu'))
    input = output2[i]
  
  return output2[5];

def candydisc(input, reuse=False, is_training=True, name='disconv'):
  k=32
  for i in range(0,7):
    if i%2==0:
      k=k*2
    if i==0:
      output = conv3_3(input,'discconv3x3_{}'.format(i), reuse=reuse,is_training=is_training, norm='none',activation='LeakyReLU',k=k,strides=2)
    elif i==6:
      output = conv3_3(input,'discconv3x3_{}'.format(i), reuse=reuse,is_training=is_training,norm='none', activation='sigmoid',k=1,strides=2)
    else:
      output = conv3_3(input,'discconv3x3_{}'.format(i), reuse=reuse,is_training=is_training, activation='LeakyReLU',k=k,strides=2)
    input = output
  
  return output;

  
def conv3_3(input,name, reuse=False, is_training=True, activation='Prelu',norm='batch',prev_input=None, k=32,strides=1):
  """ A 7x7 Convolution-BatchNorm-ReLU layer with k filters and stride 1
  Args:
    input: 4D tensor
    k: integer, number of filters (output depth)
    norm: 'instance' or 'batch' or None
    activation: 'relu' or 'tanh'
    name: string, e.g. 'c7sk-32'
    is_training: boolean or BoolTensor
    name: string
    reuse: boolean
  Returns:
    4D tensor
  """
  with tf.variable_scope(name, reuse=reuse):
    weights = _weights("weights",
      shape=[3, 3, input.get_shape()[3], k])

    # padding only width and height of 3 rows each.
    # input: batch_size x width x height x 3

    padded = tf.pad(input, [[0,0],[1,1],[1,1],[0,0]], 'CONSTANT')
    if strides ==1:
      conv = tf.nn.conv2d(padded, weights,strides=[1, 1, 1, 1], padding='VALID')
    else:
      conv = tf.nn.conv2d(padded, weights,strides=[1, 2, 2, 1 ], padding='VALID')

    normalized=conv
    if norm =='batch':
      normalized = _batch_norm(conv, is_training)
    if prev_input is not None:
      normalized=normalized+prev_input
    if activation == 'Prelu':
      output = parametric_relu(normalized)
    if activation == 'relu':
      output = tf.nn.relu(normalized)
    if activation == 'tanh':
      output = tf.nn.tanh(normalized)
    if activation == 'LeakyReLU':
      output = tf.nn.leaky_relu(normalized,alpha=0.2)
    if activation == 'sigmoid':
      output = tf.nn.sigmoid(normalized)
    return output

def parametric_relu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5

  return pos + neg

### Helpers
def _weights(name, shape, mean=0.0, stddev=0.02):
  """ Helper to create an initialized Variable
  Args:
    name: name of the variable
    shape: list of ints
    mean: mean of a Gaussian
    stddev: standard deviation of a Gaussian
  Returns:
    A trainable variable
  """
  var = tf.get_variable(
    name, shape,
    initializer=tf.random_normal_initializer(
      mean=mean, stddev=stddev, dtype=tf.float32))
  return var

def _leaky_relu(input, slope):
  return tf.maximum(slope*input, input)

def _batch_norm(input, is_training):
  """ Batch Normalization
  """
  # with tf.variable_scope("batch_norm"):
  #   return tf.contrib.layers.batch_norm(input,
  #                                       decay=0.9,
  #                                       scale=True,
  #                                       # updates_collections=None,
  #                                       is_training=is_training)

  with tf.variable_scope("batch_norm"):
    depth = input.get_shape()[3]
    scale = _weights("scale", [depth], mean=1.0)
    offset = _biases("offset", [depth])
    mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    epsilon = 1e-5
    inv = tf.rsqrt(variance + epsilon)
    normalized = (input-mean)*inv
    return scale*normalized + offset  

def _biases(name, shape, constant=0.0):
  """ Helper to create an initialized Bias with constant
  """
  return tf.get_variable(name, shape,
            initializer=tf.constant_initializer(constant))

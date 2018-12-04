import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # Images are 100x196 pixels, and have one color channel
  input_layer = tf.reshape(features["x"], [-1, 100, 195, 1])

  bn_1 = tf.layers.batch_normalization(input_layer, 1)
  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 100, 196, 32]
  # Output Tensor Shape: [batch_size, 100, 196, 64]
  conv1 = tf.layers.conv2d(
      inputs=bn_1,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 100, 196, 32]
  # Output Tensor Shape: [batch_size, 50, 98, 32]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)

  bn_2 = tf.layers.batch_normalization(pool1, 1)
  # Convolutional Layer #2
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 50, 98, 32]
  # Output Tensor Shape: [batch_size, 50, 98, 64]
  conv2 = tf.layers.conv2d(
      inputs=bn_2,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 50, 98, 64]
  # Output Tensor Shape: [batch_size, 25, 49, 64]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1)

  bn_3 = tf.layers.batch_normalization(pool2, 1)

  # Convolutional Layer #3
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size, 25, 49, 128]
  # Output Tensor Shape: [batch_size,25, 49, 128]
  conv3 = tf.layers.conv2d(
      inputs=bn_3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #3
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size,25, 49, 128]
  # Output Tensor Shape: [batch_size,12.5, 24.5, 128]
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=1)


  bn_4 = tf.layers.batch_normalization(pool3, 1)

  # Convolutional Layer #4
  # Computes 64 features using a 5x5 filter.
  # Padding is added to preserve width and height.
  # Input Tensor Shape: [batch_size,12.5, 24.5, 128]
  # Output Tensor Shape: [batch_size,12.5, 24.5, 256]
  conv4 = tf.layers.conv2d(
      inputs=bn_4,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #4
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size,12.5, 24.5, 256]
  # Output Tensor Shape: [batch_size,6.25, 12.25, 256]
  pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=1)

  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool4_flat = tf.reshape(pool4, [-1, 6.25 * 12.25 * 256])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size,  6.25 * 12.25 * 256]
  # Output Tensor Shape: [batch_size, 4096]
  dense = tf.layers.dense(inputs=pool4_flat, units=4096, activation=tf.nn.relu)
  dense_2 = tf.layers.dense(inputs=dense, units=4096, activation=tf.nn.relu)

  # Output Tensor Shape: [batch_size, 4096]
  # Output Tensor Shape: [batch_size, 468]
  dense_3 = tf.layers.dense(inputs=dense_2, units=468, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense_3, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 36]
  logits = tf.layers.dense(inputs=dropout, units=36)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
      x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
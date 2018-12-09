import numpy as np
import os
import cv2
import re
import pandas as pn
import tensorflow as tf
import glob

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # Images are 100x196 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 196, 100, 1])

    bn_1 = tf.layers.batch_normalization(input_layer, 1)
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 100, 196, 32]
    # Output Tensor Shape: [batch_size, 100, 196, 32]
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

    drop_1 = tf.layers.dropout(pool1, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    bn_2 = tf.layers.batch_normalization(drop_1, 1)
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

    drop_2 = tf.layers.dropout(pool2, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    bn_3 = tf.layers.batch_normalization(drop_2, 1)

    # Convolutional Layer #3
    # Computes 128 features using a 5x5 filter.
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

    drop_3 = tf.layers.dropout(pool3, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    bn_4 = tf.layers.batch_normalization(drop_3, 1)

    # Convolutional Layer #4
    # Computes 256 features using a 5x5 filter.
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

    drop_4 = tf.layers.dropout(pool4, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 6.25, 12.25, 256]
    # Output Tensor Shape: [batch_size, 6.25 * 12.25 * 256]
    pool4_flat = tf.reshape(drop_4, [-1, 6.25 * 12.25 * 256])

    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size,  6.25 * 12.25 * 256]
    # Output Tensor Shape: [batch_size, 4096]
    dense = tf.layers.dense(inputs=pool4_flat, units=4096, activation=tf.nn.relu)
    drop_fc_1 = tf.layers.dropout(dense, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    dense_2 = tf.layers.dense(inputs=drop_fc_1, units=4096, activation=tf.nn.relu)
    drop_fc_2 = tf.layers.dropout(dense_2, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Output Tensor Shape: [batch_size, 4096]
    # Output Tensor Shape: [batch_size, 468]
    dense_3 = tf.layers.dense(inputs=drop_fc_2, units=468, activation=tf.nn.softmax)

    # Add dropout operation; 0.25 probability that element will be kept
    dropout = tf.layers.dropout(inputs=dense_3, rate=0.25, training=mode == tf.estimator.ModeKeys.TRAIN)

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


def images_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


# takes in a folder path and goes through it and returns the images in it that match the allowed types of images
def image_array(folder_path):
    array = list()
    for class_dir in os.listdir(folder_path):
        for image_path in glob.glob(class_dir):
            im = cv2.imread(image_path)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            array.append(gray)
    return np.asarray(array, dtype=np.int32)

def one_hot_encoding(csv_labels):
    value_set = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    char_to_int = dict((c, i) for i, c in enumerate(value_set))
    labels = list()
    for label in csv_labels:
        int_encoded = [char_to_int[char] for char in label]
        one_hot_encoded = list()
        for value in int_encoded:
            letter = [0 for _ in range(len(value_set))]
            letter[value] = 1
            one_hot_encoded.append(letter)
        labels.append(one_hot_encoded)
    return np.asanyarray(labels, dtype=np.int32)


def main(unused_argv):
    # Load training and eval data
    train_image_dir = input('training data dir path: ')
    train_data = image_array(train_image_dir)
    csv_file_dir = input('training data labels path: ')
    csv_data = pn.read_csv(csv_file_dir)
    train_labels = one_hot_encoding(csv_data['Barcode_Value'])
    # eval_data = mnist.test.images  # Returns np.array
    # eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Create the Estimator
    barcode_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/barcode_convnet_model")

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    barcode_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    # eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
    #   x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
    # val_results = barcode_classifier.evaluate(input_fn=eval_input_fn)
    # print(eval_results)


if __name__ == "__main__":
    tf.app.run()

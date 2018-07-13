# The TensorFlow layers module provides a high-level API that makes it easy to
#  construct a neural network
# It provides methods that facilitate the creation of dense (fully connected)
# layers and convolutional layers, adding activation functions, and applying
# dropout regularization

# MNIST comprises 60,000 training examples and 10,000 test examples of the
# handwritten digits 0–9, formatted as 28x28-pixel monochrome images

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import  numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    """Model function for CNN.""" # this is docstring and is used as documentation
    # Input layer
    input_layer=tf.reshape(features["x"], [-1, 28, 28, 1])
    # layer model expects input tensors to have a shape of [batch_size, image_height, image_width, channels]
    # channel: Number of color channels in the example images. For color images, the number of channels is 3 (red, green, blue). For monochrome images, there is just 1 channel (black)
    # batch size is set to 1, which specifies this dimension should be dynamically computed based on the number of input values in feature["x"], holding the size of all other dimensions constant. This allows us to treat batch_size as a hyperparameter that we can tune.
    # For example, if we feed examples into our model in batches of 5, feature["x"] will contain 3920 values (one value for each pixel in each image), and input layer will have a shape of [5, 28, 28, 1]

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer, # input tensor, which must have the shape [batch_size, image_height, image_width, channels]
        filters=32, # number of filters in the convolution (the dimensionality of
                    #  the output space)
        kernel_size=[5,5], # size (dimension) of filter # [height, width]
        padding="same", # valid: without padding-drop whatever remained on the right
                        # same: with zero padding-try to pad evenly left and right
                        # same: add 0 values to the edges of input tensor to preserve height and width of 28 (otherwise, 5*5 convlution over 28*28 tensor will produce a 24*24 tensor)
        activation=tf.nn.relu) # the activation function that applies to the output of the convolution
    # conv1 has a shape of [batch_size, 28, 28, 32]

    #Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2) # for different stride values for height and width, can use stride=[3, 6]
    # pool1 has a shape of [batch_size, 14, 14, 32]



    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    # conv2 has a shape of [batch_size, 14, 14, 64] # 64 channels for the 64 filters applied

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # pool2 has a shape of [batch_size, 7, 7, 64]]


    # Dense Layer
    # add a dense layer (with 1024 neurons and ReLU activation)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64]) # flatten feature map to shape [batch_size, features] so that tensor has only 2 dimensions
    # again the -1 signifies batch_size dimension will be dynamically calculated based on the number of examples in the input data
    # pool2_flat has shape [batch_size, 3136]

    # now use dense() to connect dense layer
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # units: number of neurons in the dense layer
    dropout = tf.layers.dropout( # apply dropout regularization to dense layer to improve results
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    # dropout rate: percent of elements randomly dropped during training
    # training: takes a boolean specifying whether the model is currently being run in the training mode, dropout only performed if training is True
    # dropout has shape [batch_size, 1024] # following number of neurons in dense layer

    # Logits Layer
    # return the raw values for predictions; consists of another dense layer with 10 neurons, one for each target class digit 0-9

    logits = tf.layers.dense(inputs=dropout, units=10) # activation is default (linear activation)
    # logits has shape [batch_size, 10]



    # Generate Predictions
    # convert raw values tensor into two different formats for model function to return:
    # 1. The predicted class for each example: a digit from 0-9
    # 2. The probabilities for each possible target class for each example

    # For a given example, predicted calss is the element in the corresponding row of the logits tensor with the highest raw value
    # The index of this element is found using tf.argmax

    predictions={
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # input: input tensor from which to extract max values
        # axis: specifies the axis of the input tensor along which to find the max value; here find along dimension with index 1

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



# Training and Evaluating the CNN MNIST Classifier

def main(unused_argv):
    # Load training and evaluation data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # training feature data: raw pixel values for 55000 images of hand-drawn digits
    # training labels: corresponding digit from 0-9 for each image
    # evaluation feature data: raw pixel values for 10000 images
    # evaluation labels: ground truth digits

    # Create the Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")
    # model_fn: specifies the model function to use for training, evaluation and predicition
    # model_dir: specifies t he directory where model data will be saved; here the directory is /tmp/mnist_convnet_model but can change to others



    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    # store a dict of the tensors we want to log in tensor_to_log; each key is a label of our choice that will be printed in the log output, and the corresponding label is the name of a tensor
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)
    # tf.train.LoggingTensorHook: prints the given tensors every N local steps, every N seconds, or at end
    # The tensors will be printed to the log, with INFO severity



    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        # pass training feature data and labels to x (as a dict) and y
        x={"x": train_data},
        y=train_labels,
        batch_size=100, # model trains on minibatches of 100 examples at each step
        num_epochs=None, # model will train until the specified number of steps is reach
        shuffle=True) # shuffle the training data

    mnist_classifier.train(
        input_fn=train_input_fn, # pass in input function
        steps=20000, # pass in training steps
        hooks=[logging_hook]) # pass in logging_hook which is to be triggered during training



    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1, # model evaluates the metrics over one epoch of data and return the result
        shuffle=False) # no need to shuffle data; iterate through the data sequentially

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn) # pass in input function
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()










# Intro to CNN

# CNNs apply a series of filters to the raw pixel data of an image to extract and
# learn high level features, which the model can then use for classification.

# It contains three components:

# Convolutional layers, which apply a specified number of convlution filters to the
# image. For each subregion, the layer performs a set of mathematical operations to
# produce a single value in the output feature map.
# Convolutional layers then typically apply a ReLU activation function to the output
# to introduce nonlinearities into the model

# Pooling layers, which downsample the image data extracted by the convolutional layers
# to reduce the dimensionality of the feature map, in order to decrease processing time.
# A commonly used pooling algorithm is max pooling, which extracts subregions of the
# feature map (e.g. 2*2-pixel tiles), keep their max value, and discard all other values.

# Dense layers (fully connected layers), whcih perform classification on the features
# extracted by the convolutional layers and downsampled by the pooling layers. In a dense
# layer, every node in the layer is connected to every node in the preceding layer.

# Typically, a CNN is composed of a stack of convolutional modules that perform feature
# extraction. Each module consists of a convolutional layer followed by a pooling layer.
# The last convolutional module is followed by one or more dense layers that perform
# classification. The final dense layer in a CNN contains a single node for each target
# class in the model (all the possible classes the model may predict), with a softmax
# activation function to generate a value between 0-1 for each node (the sum of all these
# softmax values is equal to 1). We can interpret the softmax values for a given image as
# relative measurements of how likely it is that the image falls into each target class.


# Building the CNN MNIST Classifier

# CNN architecture:
# 1. Convolutional Layer #1: Applies 32 5x5 filters (extracting 5x5-pixel subregions),
# with ReLU activation function
# 2. Pooling Layer #1: Performs max pooling with a 2x2 filter and stride of 2 (which
# specifies that pooled regions do not overlap)
# 3. Convolutional Layer #2: Applies 64 5x5 filters, with ReLU activation function
# 4. Pooling Layer #2: Again, performs max pooling with a 2x2 filter and stride of 2
# 5. Dense Layer #1: 1,024 neurons, with dropout regularization rate of 0.4 (probability
#  of 0.4 that any given element will be dropped during training)
# 6. Dense Layer #2 (Logits Layer): 10 neurons, one for each digit target class (0–9)


# tf.layers module
# contains methods to create each of the three layer types above
# conv2d(): constructs a 2d convolutional payer.
# max_pooling2d(): constructs a 2d pooling layer using max-pooling algorithm
# dense(): constructs a dense layer
# Each of these methods accept a tensor as input and returns a transformed tensor as output
# This makes it easy to connect one layer to another: just take the output from  one layer-
# creation method and supply it as input to another


# Note: training CNNs is quite computationally intensive. To train more quickly, decrease
# the number of steps passed to train(), but this will affect accuracy
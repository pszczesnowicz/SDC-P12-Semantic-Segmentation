import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'),\
    'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# # Check for a GPU
# if not tf.test.gpu_device_name():
#     warnings.warn('No GPU found. Please use a GPU to train your neural network.')
# else:
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()

    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """

    # Encoder

    # Layer 7 1x1 convolution layer: Output 5x18x2
    layer7_conv1x1 = tf.layers.conv2d(vgg_layer7_out, filters=num_classes, kernel_size=1, strides=(1, 1),
                                      padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Layer 4 1x1 convolution layer: Output 10x36x2
    layer4_conv1x1 = tf.layers.conv2d(vgg_layer4_out, filters=num_classes, kernel_size=1, strides=(1, 1),
                                      padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Layer 3 1x1 convolution layer: Output 20x72x2
    layer3_conv1x1 = tf.layers.conv2d(vgg_layer3_out, filters=num_classes, kernel_size=1, strides=(1, 1),
                                      padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Decoder

    # Layer 7 transpose convolution layer (2x up-sampling): Output 10x36x2
    output1 = tf.layers.conv2d_transpose(layer7_conv1x1, filters=num_classes, kernel_size=4, strides=(2, 2),
                                         padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Layer 7 skip connections with layer 4: Output 10x36x2
    input2 = tf.add(output1, layer4_conv1x1)

    # Layers 7 and 4 transpose convolution layer (2x up-sampling): Output 20x72x2
    output2 = tf.layers.conv2d_transpose(input2, filters=num_classes, kernel_size=4, strides=(2, 2), padding='same',
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    # Layers 7 and 4 skip connections with layer 3: Output 20x72x2
    input3 = tf.add(output2, layer3_conv1x1)

    # Layers 7, 4, and 3 transpose convolution layer (8x up-sampling): Output 160x576x2
    nn_last_layer = tf.layers.conv2d_transpose(input3, filters=num_classes, kernel_size=16, strides=(8, 8),
                                               padding='same', kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))

    return nn_last_layer


tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # Reshaping the tensor from 4D to 2D
    logit = tf.reshape(nn_last_layer, (-1, num_classes))

    # Reshaping the labels from
    label = tf.reshape(correct_label, (-1, num_classes))

    # Calculating the mean probability of error
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))

    # Reducing the mean probability of error by optimizing the weights and biases
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logit, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):

        for image, label in get_batches_fn(batch_size):
            sess.run([train_op, cross_entropy_loss],
                     feed_dict={input_image: image, correct_label: label, keep_prob: 0.5, learning_rate: 1e-3})

    return None


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    epochs = 1
    batch_size = 4

    # Download pre-trained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # Batch variables
    correct_label = tf.placeholder(tf.int32, (None, None, None, num_classes))
    learning_rate = tf.placeholder(tf.float32, None)

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Build NN using load_vgg, layers, and optimize function
        input_image, vgg_keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        saver = tf.train.Saver()

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, vgg_keep_prob, learning_rate)

        saver.save(sess, './model/fcn')

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
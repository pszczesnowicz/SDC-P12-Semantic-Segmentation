import os.path
import tensorflow as tf
import numpy as np
import helper
import time
from distutils.version import LooseVersion
# import project_tests as tests
# import warnings

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


# tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, init_stddev, reg_scale):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :param init_stddev:
    :param reg_scale:
    :return: The Tensor for the last layer of output
    """

    # Layer 7 1x1 convolution layer: Output 5x18x2
    layer7_conv1x1 = tf.layers.conv2d(vgg_layer7_out, filters=num_classes, kernel_size=1, strides=(1, 1),
                                      padding='same',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=init_stddev),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                      name='layer7_conv1x1')

    # Layer 4 1x1 convolution layer: Output 10x36x2
    layer4_conv1x1 = tf.layers.conv2d(vgg_layer4_out, filters=num_classes, kernel_size=1, strides=(1, 1),
                                      padding='same',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=init_stddev),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                      name='layer4_conv1x1')

    # Layer 3 1x1 convolution layer: Output 20x72x2
    layer3_conv1x1 = tf.layers.conv2d(vgg_layer3_out, filters=num_classes, kernel_size=1, strides=(1, 1),
                                      padding='same',
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=init_stddev),
                                      kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                      name='layer3_conv1x1')

    # Layer 7 transpose convolution layer (2x up-sampling): Output 10x36x2
    output1 = tf.layers.conv2d_transpose(layer7_conv1x1, filters=num_classes, kernel_size=4, strides=(2, 2),
                                         padding='same',
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=init_stddev),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                         name='output1')

    # Layer 7 skip connections with layer 4: Output 10x36x2
    input2 = tf.add(output1, layer4_conv1x1, name='input2')

    # Layers 7 and 4 transpose convolution layer (2x up-sampling): Output 20x72x2
    output2 = tf.layers.conv2d_transpose(input2, filters=num_classes, kernel_size=4, strides=(2, 2),
                                         padding='same',
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=init_stddev),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                         name='output2')

    # Layers 7 and 4 skip connections with layer 3: Output 20x72x2
    input3 = tf.add(output2, layer3_conv1x1, name='input3')

    # Layers 7, 4, and 3 transpose convolution layer (8x up-sampling): Output 160x576x2
    output3 = tf.layers.conv2d_transpose(input3, filters=num_classes, kernel_size=16, strides=(8, 8),
                                         padding='same',
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=init_stddev),
                                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                         name='output3')

    return output3


# tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, num_classes, learning_rate, beta1, beta2, epsilon):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param num_classes: Number of classes to classify
    :param learning_rate: TF Placeholder for the learning rate
    :param beta1:
    :param beta2:
    :param epsilon:
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    # Reshaping the tensor from 4D to 2D
    logit = tf.reshape(nn_last_layer, (-1, num_classes), name='logit')
    label = tf.reshape(correct_label, (-1, num_classes), name='label')

    # Calculating the regularization loss
    regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='reg_loss')

    # Calculating the cross entropy loss
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit),
                                        name='ce_loss')

    total_loss = regularization_loss + cross_entropy_loss

    # Reducing the total loss by optimizing
    train_op = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(total_loss)

    return logit, train_op, total_loss


# tests.test_optimize(optimize)


def run():

    data_dir = './data'
    runs_dir = './runs'
    models_dir = './models'

    # tests.test_for_kitti_dataset(data_dir)

    # Download pre-trained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')

    image_shape = (160, 576)

    # Parameters
    num_classes = 2
    epochs = 10
    batch_size = 4

    # Layer hyperparameters:
    # Kernel initializer standard deviations
    init_stddevs = np.array([1e-2])
    # Kernel regularizer scales
    reg_scales = np.array([1e-3])

    # Adam optimizer hyperparameters:
    # Learning rates
    learning_rates = np.array([1e-3])
    # Beta1s
    beta1s = np.array([0.9])
    # Beta2s
    beta2s = np.array([0.999])
    # Epsilons
    epsilons = np.array([1e-8])

    # Create function to get batches
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

    for init_stddev in init_stddevs:

        for reg_scale in reg_scales:

            for learning_rate in learning_rates:

                for beta1 in beta1s:

                    for beta2 in beta2s:

                        for epsilon in epsilons:

                            model_name = 'model_' + str(init_stddev) + '_' + str(reg_scale) + '_' +\
                                         str(learning_rate) + '_' + str(beta1) + '_' + str(beta2) + '_' + str(epsilon)

                            save_dir = os.path.join(models_dir, model_name)

                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)

                            tf.reset_default_graph()

                            with tf.Session() as sess:

                                # Variables
                                correct_label = tf.placeholder(tf.int32, (None, None, None, num_classes),
                                                               name='correct_label')

                                # Build FCN decoder using layers extracted from VGG Net
                                input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out =\
                                    load_vgg(sess, vgg_path)

                                # Build FCN encoder
                                nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes,
                                                       init_stddev, reg_scale)

                                # Set up optimizer
                                logits, train_op, total_loss = optimize(nn_last_layer, correct_label, num_classes,
                                                                        learning_rate, beta1, beta2, epsilon)

                                sess.run(tf.global_variables_initializer())
                                saver = tf.train.Saver()

                                best_loss = 1

                                print('Layer Hyperparameters:')
                                print('Kernel initializer standard deviation  = {:.3e}'.format(init_stddev))
                                print('Kernel regularizer scale = {:.3e}'.format(reg_scale))
                                print()

                                print('Adam Optimizer Hyperparameters:')
                                print('Learning rate = {:.3e}'.format(learning_rate))
                                print('Beta 1 = {:.4f}'.format(beta1))
                                print('Beta 2 = {:.4f}'.format(beta2))
                                print('Epsilon = {:.3e}'.format(epsilon))
                                print()

                                print('Training...')
                                print()

                                for epoch in range(epochs):

                                    t0 = time.time()

                                    for image, label in get_batches_fn(batch_size):

                                        _, loss = sess.run([train_op, total_loss],
                                                           feed_dict={input_image: image, correct_label: label,
                                                                      keep_prob: 0.5})

                                    t1 = time.time()

                                    print('Epoch {}'.format(epoch + 1))
                                    print('Training time = {:.3f} s'.format(t1 - t0))
                                    print('Loss = {:.3f}'.format(loss))

                                    if loss < best_loss:
                                        best_loss = loss
                                        saver.save(sess, save_dir + '/fcn')

                                        print('Loss improved: model saved \n')
                                        print()

                                    else:
                                        print('Loss did not improve: model not saved \n')
                                        print()

                                # Save inference data using helper.save_inference_samples
                                helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits,
                                                              keep_prob, input_image)


if __name__ == '__main__':
    run()

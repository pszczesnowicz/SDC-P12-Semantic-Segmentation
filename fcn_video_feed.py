import main
import os.path
import tensorflow as tf
import scipy.misc
import numpy as np
# Uncomment the following two lines if first time running
# import imageio
# imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip


def segment(clip):

    # Resize 720x1280x3 video clip to 736x1280x3 image to prevent shape mismatch due to rounding up
    image = scipy.misc.imresize(clip, image_shape)

    im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, input_image: [image]})

    im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])

    segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
    
    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))

    mask = scipy.misc.toimage(mask, mode="RGBA")

    street_im = scipy.misc.toimage(image)

    street_im.paste(mask, box=None, mask=mask)

    return np.array(street_im)


# Image information
num_classes = 2
image_shape = (736, 1280)

# TensorFlow placeholders
correct_label = tf.placeholder(tf.int32, (None, None, None, num_classes))
learning_rate = tf.placeholder(tf.float32, None)

with tf.Session() as sess:

    # Path to vgg model
    vgg_path = os.path.join('./data', 'vgg')

    # Build FCN model using load_vgg, layers, and optimize functions
    input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = main.load_vgg(sess, vgg_path)

    nn_last_layer = main.layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

    logits, train_op, cross_entropy_loss = main.optimize(nn_last_layer, correct_label, learning_rate, num_classes)

    # Create saver
    saver = tf.train.Saver()

    # Restore trained FCN model
    saver.restore(sess, './model/model_0.01_0.001_0.001_0.9_0.999_1e-8/fcn')

    # Feed video to FCN model
    video_input = VideoFileClip('./videos/input/test_video.mp4')
    video_output = './videos/output/test_video.mp4'
    video_clip = video_input.fl_image(segment)
    video_clip.write_videofile(video_output, audio=False)

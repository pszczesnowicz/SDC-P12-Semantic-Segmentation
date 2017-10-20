import main
import os.path
import tensorflow as tf
import scipy.misc
import numpy as np
import imageio
imageio.plugins.ffmpeg.download()
from moviepy.editor import VideoFileClip


def paint(clip):

    image = scipy.misc.imresize(clip, gl_image_shape)

    im_softmax = gl_sess.run([tf.nn.softmax(gl_logits)], {gl_keep_prob: 1.0, gl_input_image: [image]})

    im_softmax = im_softmax[0][:, 1].reshape(gl_image_shape[0], gl_image_shape[1])

    segmentation = (im_softmax > 0.5).reshape(gl_image_shape[0], gl_image_shape[1], 1)

    mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))

    mask = scipy.misc.toimage(mask, mode="RGBA")

    street_im = scipy.misc.toimage(image)

    street_im.paste(mask, box=None, mask=mask)

    return np.array(street_im)


num_classes = 2
image_shape = (736, 1280)  # (160, 576)
data_dir = './data'
runs_dir = './runs'

# Placeholders
correct_label = tf.placeholder(tf.int32, (None, None, None, num_classes))
learning_rate = tf.placeholder(tf.float32, None)

with tf.Session() as sess:
    # Path to vgg model
    vgg_path = os.path.join(data_dir, 'vgg')

    # Build NN using load_vgg, layers, and optimize function
    input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = main.load_vgg(sess, vgg_path)

    nn_last_layer = main.layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

    logits, train_op, cross_entropy_loss = main.optimize(nn_last_layer, correct_label, learning_rate, num_classes)

    saver = tf.train.Saver()

    saver.restore(sess, './model/fcn')

    # helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, vgg_keep_prob, input_image)

    global gl_sess, gl_logits, gl_keep_prob, gl_input_image, gl_image_shape

    gl_sess = sess
    gl_logits = logits
    gl_keep_prob = keep_prob
    gl_input_image = input_image
    gl_image_shape = image_shape

    video_input = VideoFileClip('./videos/input/test_video.mp4')
    video_output = './videos/output/project_video.mp4'
    video_clip = video_input.fl_image(paint)
    video_clip.write_videofile(video_output, audio=False)

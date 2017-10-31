import re
import numpy as np
import os.path
import scipy.misc
from glob import glob

data_dir = 'data/data_road/training'

image_paths = glob(os.path.join(data_dir, 'image_2', '*.png'))
label_paths = {re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
               for path in glob(os.path.join(data_dir, 'gt_image_2', '*_road_*.png'))}

count = 0

for image in image_paths:

    # Flip image and ground truth image horizontally
    new_image = np.flip(scipy.misc.imread(image), axis=1)
    new_label = np.flip(scipy.misc.imread(label_paths[os.path.basename(image)]), axis=1)

    count += 1

    new_image_name = os.path.splitext(image)[0] + str(count) + '.png'
    new_label_name = os.path.splitext(label_paths[os.path.basename(image)])[0] + str(count) + '.png'

    scipy.misc.toimage(new_image).save(new_image_name)
    scipy.misc.toimage(new_label).save(new_label_name)

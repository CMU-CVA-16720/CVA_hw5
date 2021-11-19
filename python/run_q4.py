import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir('../images'):
    # Get image and bounding boxes
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
    bboxes, bw = findLetters(im1)
    
    # Display image + bboxes
    if False:
        plt.imshow(bw)
        for bbox in bboxes:
            minr, minc, maxr, maxc = bbox
            rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
        plt.show()

    # Get median height
    height_array = []
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        height_array.append(maxr-minr)
    height_median = np.median(height_array)
    
    # Compute rows
    rows = []
    cur_row = []
    last_minr = 0
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        # First letter special case
        if(last_minr == 0):
            # Put first letter in row and move on
            cur_row.append(bbox)
            last_minr = minr
            continue
        delta_minr = abs(minr - last_minr)
        # New row if delta(minr) exceeds median height
        if(delta_minr > height_median):
            # New row
            rows.append(cur_row)
            cur_row = []
            cur_row.append(bbox)
        else:
            # Continue current row
            cur_row.append(bbox)
        last_minr = minr
    # Append last row
    rows.append(cur_row)

    # Sort within rows for correct ordering
    for row in rows:
        row.reverse()
    # Crop, tranpose and flatten
    for row in rows:
        for bbox in row:
            minr, minc, maxr, maxc = bbox
            plt.imshow(bw[minr:maxr,minc:maxc])
            plt.show()
    print("blah")



    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    
    # # load the weights
    # # run the crops through your neural network and print them out
    # import pickle
    # import string
    # letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    # params = pickle.load(open('q3_weights.pickle','rb'))
    # ##########################
    # ##### your code here #####
    # ##########################
    
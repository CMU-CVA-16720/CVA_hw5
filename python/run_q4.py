import os
from matplotlib import scale
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
import skimage.transform

from nn import *
from q4 import *
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from math import floor, ceil
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
        # Most elements are in reverse order
        row.reverse()
        row.sort(key = lambda x: x[1])



    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    x_matrix = []
    x_vect = []
    for row in rows:
        for bbox in row:
            # Get bounding box location and dimensions
            minr, minc, maxr, maxc = bbox
            width, height = maxc-minc, maxr-minr
            wh_ratio = width/height
            # Get and manipulate image
            cur_img_org = bw[minr:maxr,minc:maxc]
            if(wh_ratio < 1):
                # Bbox is tall
                scaled_width = floor(28*wh_ratio)
                cur_img_rshp = skimage.transform.resize(cur_img_org,(28,scaled_width))
                # Padding
                width_padding = 2+(28-scaled_width)/2
                cur_img_rshp = np.pad(cur_img_rshp,((2,2),(floor(width_padding),ceil(width_padding))))
                pass
            else:
                # Bbox is wide
                scaled_height = floor(28/wh_ratio)
                cur_img_rshp = skimage.transform.resize(cur_img_org,(scaled_height,28))
                # Padding
                height_padding = 2+(28-scaled_height)/2
                cur_img_rshp = np.pad(cur_img_rshp,((floor(height_padding),ceil(height_padding)),(2,2)))
                pass
            cur_img = 1-cur_img_rshp
            # Turn into vector, then append
            x = np.transpose(cur_img).flatten()
            x_vect.append(x)
        x_matrix.append(x_vect)
        x_vect = []
    
    # # load the weights
    # # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    class_to_letter = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    for x_row in x_matrix:
        cur_row = ''
        for xb in x_row:
            # forward
            h1 = forward(np.expand_dims(xb,axis=0),params,'layer1')
            probs = forward(h1,params,'output',softmax)
            predic = np.argmax(probs)
            cur_row += class_to_letter[predic]
            # # Debugging
            # print("Classification = {}".format(class_to_letter[predic]))
            # plt.imshow(np.transpose(np.reshape(xb,(32,32))))
            # plt.show()
        print(cur_row)
    print()
    
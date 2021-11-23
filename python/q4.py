import numpy as np

import skimage
# import skimage.measure
# import skimage.color
# import skimage.restoration
# import skimage.filters
# import skimage.morphology
# import skimage.segmentation

from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb, rgb2gray
from skimage.restoration import denoise_bilateral

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image, display=False, minsize = 100):
    # Output setup
    bboxes = []
    bw = None
    # denoise
    image_clean = denoise_bilateral(image, multichannel = True)
    # convert to grayscale
    image_gray = skimage.color.rgb2gray(image_clean)
    # blur
    image_blur = skimage.filters.gaussian(image_gray, 2)
    # apply threshold
    thresh = threshold_otsu(image_blur)
    bw = closing(image_gray < thresh, square(10))
    # remove artifacts connected to image border
    cleared = clear_border(bw)
    # label image regions
    label_image = label(cleared)
    for i,region in enumerate(regionprops(label_image)):
        # take regions with large enough areas
        if region.area >= minsize:
            # print("{}: Region size: {}".format(i+1,region.area))
            bboxes.append(region.bbox)
    # Display image with bounding boxes
    if(display):
        # to make the background transparent, pass the value of `bg_label`,
        # and leave `bg_color` as `None` and `kind` as `overlay`
        image_label_overlay = label2rgb(label_image, image=image, bg_label=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        # ax.imshow(label_image)
        ax.imshow(image_label_overlay)
        for ind, cur_bbox in enumerate(bboxes):
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = cur_bbox
            print("{} = {} x {}".format(ind, maxr-minr, maxc-minc))
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                    fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
        print("Total bboxes = {}".format(len(bboxes)))
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()
    return bboxes, bw.astype(float)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import os
    import skimage.io
    for img in os.listdir('../images'):
        im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
        bboxes, bw = findLetters(im1, True)

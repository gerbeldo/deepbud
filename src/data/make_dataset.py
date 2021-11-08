import os
import pickle as pkl
from random import randint

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io as io
import skimage.measure
from scipy.ndimage.morphology import binary_fill_holes

from definitions import *


def list_img(dir, pat):
    """Returns list containing (sorted) filenames that start with pat"""
    files = os.listdir(dir)

    a = []

    for file in files:
        if file.startswith(pat):
            a.append(file)
        else:
            pass

    a.sort()

    return a


def list_img_masks(dir, img_pat="img", msk_pat="msk"):
    """Returns list with all image and corresponding mask names"""

    imgs = list_img(dir, img_pat)
    masks = list_img(dir, msk_pat)

    return [imgs, masks]


def cell_mask(img):
    """Returns list containing the contour and fill of CellID segmentation"""

    contour = img == 255
    union = binary_fill_holes(contour)
    fill = np.logical_xor(contour, union)

    cellid_mask = (contour, fill, union)

    return cellid_mask


def square(x, y, boxsize, pad=0):
    """given (x,y), generate a square of boxsize lenght and center (x,y).
    Returns tuple containing xmin, xmax, ymin, ymax

    If using skimage.measure.label regionprops.centroid, x = col, y = row!
    It's backwards!
    """

    r = boxsize / 2

    x0 = x - r + pad
    x1 = x + r + pad
    y0 = y - r + pad
    y1 = y + r + pad

    return (x0, x1, y0, y1)


def transf_square(coord):
    """transforms coordinates of square to xmin, ymin, width, height
    because it works better with matplotlib.patches.

    The width and height have to be equal, and to the boxsize selected."""

    x0, x1, y0, y1 = coord

    return (x0, y0, x1 - x0, y1 - y0)


# def all_squares(pos, boxsize, pad = 0):
#    '''Apply square and transf_square to every cell in position.'''
#
#     i = 0
#     lim = len(pos)
#     res = []
#     res_transf = []

#     while i < lim:

#         sq = square(pos_1.iloc[i].xpos, pos_1.iloc[i].ypos, boxsize, pad)

#         res.append(sq)
#         res_transf.append(transf_square(sq))

#         i += 1


#     return (res, res_transf)


def random_square(row, col, n):

    # list of n random squares around row, col
    res = []

    for i in range(n):

        r_row = row + (np.random.choice([-1, 1]) * np.random.randint(0, 15, 1))
        r_col = col + (np.random.choice([-1, 1]) * np.random.randint(0, 15, 1))

        res.append((int(r_row), int(r_col)))

    return res


def pad_it(img, px):
    """Add square padding of 0s around image."""

    return np.pad(img, px, mode="constant")


def plot_img_sq(img, t_sq, cmap="binary", xlim=None, ylim=None):
    """Plot img with superimposed squares surrounding cells."""

    fig, ax = plt.subplots(1, figsize=(15, 15))
    ax.imshow(img, cmap=cmap)

    for i in range(len(t_sq)):

        x, y, w, h = t_sq[i]

        ax.add_patch(
            patches.Rectangle(
                (x, y), w, h, linewidth=1, edgecolor="r", alpha=0.5
            )
        )

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.show()


def get_squares_centroids(regionprops, boxsize, pad=0, randomize=False):

    props = range(len(regionprops))
    res = []
    transf_res = []

    if randomize:
        for i in props:
            c = regionprops[i].centroid

            row, col = random_square(c[1], c[0], 1)[0]
            res.append(square(row, col, boxsize, pad))
            transf_res.append(transf_square(res[i]))

    else:
        for i in props:
            c = regionprops[i].centroid
            res.append(square(c[1], c[0], boxsize, pad))
            transf_res.append(transf_square(res[i]))

    return (res, transf_res)


def crop_img(img, squares):
    """return list with all crops of img"""

    # unpacking squares
    tot = range(len(squares))
    res = []

    for i in tot:
        # transf squares returns (x0 (col!), y0 (row!), width, height)!
        col, row, width, height = squares[i]

        row = int(round(row))
        col = int(round(col))
        width = int(round(width))
        height = int(round(height))

        crop = img[row : (row + height), col : (col + width)]

        res.append(crop)

    return res


def main(img, msk, random=False):

    # extract mask of cell interiors from CellID segmentation
    fill, union = cell_mask(msk)[1:]

    # generate label of blobs
    labels = skimage.measure.label(fill, connectivity=2)
    r_prop = skimage.measure.regionprops(labels)

    pad_union = pad_it(union, 64)
    pad_img = pad_it(img, 64)

    # generate squares
    squares = get_squares_centroids(
        r_prop, boxsize=64, pad=64, randomize=random
    )[1]

    # plot_img_sq(pad_fill, squares)

    return (crop_img(pad_img, squares), crop_img(pad_union, squares))


def gen_data(dir, img_pat="img", msk_pat="msk"):
    """Reads all images in directory that start with the pat string.
    Returns list with thumbnails, formatted like this:

    res[i][j][k]

    i: image = 0; mask = 1
    j: image
    k: thumbnail
    """

    res = [[], []]

    imgs, msks = list_img_masks(dir, img_pat, msk_pat)

    for i in range(len(imgs)):

        img = io.imread(os.path.join(dir, imgs[i]))
        msk = io.imread(os.path.join(dir, msks[i]))

        x = main(img, msk, random=True)

        res[0].append(x[0])
        res[1].append(x[1])

    return res


def plot_gen_data(a):

    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    i = randint(0, len(a))
    j = randint(0, len(a[0][i]))

    plt.subplot(131)
    plt.title("TFP")
    plt.imshow(a[0][i][j], cmap="gray")

    plt.subplot(132)
    plt.title("mask")
    plt.imshow(a[1][i][j], cmap="binary")

    plt.subplot(133)
    plt.title("merge")
    plt.imshow(a[0][i][j], cmap="gray")
    plt.imshow(a[1][i][j], cmap="Reds", alpha=0.4)

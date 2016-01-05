"""Segment image into pollen tubes."""

import argparse
import os.path
import warnings

import numpy as np

from skimage.morphology import disk

from jicbioimage.core.image import Image
from jicbioimage.core.transform import transformation
from jicbioimage.core.util.array import normalise
from jicbioimage.core.io import AutoName
from jicbioimage.transform import (
    mean_intensity_projection,
    equalize_adaptive_clahe,
    smooth_gaussian,
    find_edges_sobel,
    remove_small_objects,
    invert,
    erode_binary,
    dilate_binary,
)
from jicbioimage.illustrate import AnnotatedImage

AutoName.prefix_format = "{:02d}_"

# Suppress spurious scikit-image warnings.
warnings.filterwarnings("ignore", module="skimage.exposure._adapthist")
warnings.filterwarnings("ignore", module="skimage.util.dtype")
#warnings.filterwarnings("ignore", module="skimage.io._io")


@transformation
def identity(image):
    return image


@transformation
def threshold_gt_percentile(image, q):
    """Return image thresholdede using percentile q."""
    return image > np.percentile(image, q)


@transformation
def threshold_lt_percentile(image, q):
    """Return image thresholdede using percentile q."""
    return image < np.percentile(image, q)


@transformation
def nand_mask(im1, im2):
    """Return im1 nand im2 mask."""
    return np.logical_and(im1, np.logical_not(im2))

def highlight(intensity, mask, fname):
    red_ann = AnnotatedImage.from_grayscale(mask*255, (255, 0, 0))
    green_ann = AnnotatedImage.from_grayscale(intensity, (0, 255, 0))
    blue_ann = AnnotatedImage.from_grayscale(intensity, (0, 0, 255))
    ann = red_ann + green_ann + blue_ann
    with open(fname, "wb") as fh:
        fh.write(ann.png())
    
def get_equalised_image_and_intensity(image):
    projection = mean_intensity_projection(image)
    image = equalize_adaptive_clahe(projection)
    image = smooth_gaussian(image, sigma=1)

    intensity = normalise(invert(projection))*255
    return image, intensity

def get_edges(image):
    edges = find_edges_sobel(image)
    edges = threshold_gt_percentile(edges, 90)
    edges = remove_small_objects(edges, min_size=100)
    edges = dilate_binary(edges, selem=disk(6))
    edges = erode_binary(edges, selem=disk(6))
    edges = erode_binary(edges, selem=disk(6))
    return edges


def get_mask(image):
    mask = threshold_lt_percentile(image, 10)
    mask = remove_small_objects(mask, min_size=100)
    mask = erode_binary(mask, selem=disk(10))
    mask = dilate_binary(mask, selem=disk(10))
    mask = dilate_binary(mask, selem=disk(8))
    return mask


def get_tube(edges, mask):
    tube = nand_mask(edges, mask)
    tube = remove_small_objects(tube, min_size=700)
    return tube


def get_pollen(edges, mask):
    pollen = nand_mask(mask, edges)
    pollen = erode_binary(pollen, selem=disk(6))
    pollen = dilate_binary(pollen, selem=disk(6))
    return pollen


def get_fnames(fpath, output_dir):
    bname = os.path.basename(fpath)
    name, suffix = bname.split(".")
    if output_dir:
        name = os.path.join(output_dir, name)
    
    return "{}-pollen.png".format(name), "{}-tube.png".format(name)

def analysis(input_file, ouput_dir=None):
    pollen_fname, tube_fname = get_fnames(input_file, ouput_dir) 

    image = Image.from_file(input_file)
    image, intensity = get_equalised_image_and_intensity(image)

    edges = get_edges(image)
    mask = get_mask(image)

    tube = get_tube(edges, mask)
    pollen = get_pollen(edges, mask)

    highlight(intensity, pollen, pollen_fname)
    highlight(intensity, tube, tube_fname)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="Input jpg file")
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        parser.error("No such file: {}".format(args.input_file))

    analysis(args.input_file)


if __name__ == "__main__":
    main()

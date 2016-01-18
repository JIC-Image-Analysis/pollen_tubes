"""Segment pollen tubes."""

import os
import argparse
import warnings
import math

import numpy as np
import scipy.ndimage
from skimage.morphology import disk
import skimage.feature

from jicbioimage.core.image import Image
from jicbioimage.core.transform import transformation
from jicbioimage.core.util.color import pretty_color
from jicbioimage.core.io import AutoWrite, AutoName
from jicbioimage.transform import (
    mean_intensity_projection,
    find_edges_sobel,
    threshold_otsu,
    dilate_binary,
    erode_binary,
    remove_small_objects,
    invert,
)
from jicbioimage.segment import connected_components, watershed_with_seeds
from jicbioimage.illustrate import AnnotatedImage

# Suppress spurious scikit-image warnings.
#warnings.filterwarnings("ignore", module="skimage.exposure._adapthist")
#warnings.filterwarnings("ignore", module="skimage.util.dtype")
warnings.filterwarnings("ignore", module="skimage.io._io")


AutoName.prefix_format = "{:03d}_"


def form_factor(prop):
    return (4 * math.pi * prop.area) / float(prop.perimeter)**2


@transformation
def remove_scalebar(image, value):
    """Remove the scale bar from the image."""
    image[-42:,-154:] = value
    return image


@transformation
def threshold_abs(image, threshold):
    return image > threshold


@transformation
def remove_large_segments(segmentation, max_size):
    for i in segmentation.identifiers:
        region = segmentation.region_by_identifier(i)
        if region.area > max_size:
            segmentation[region] = 0
    return segmentation

@transformation
def remove_small_segments(segmentation, min_size):
    for i in segmentation.identifiers:
        region = segmentation.region_by_identifier(i)
        print(region.area)
        if region.area < min_size:
            segmentation[region] = 0
    return segmentation


@transformation
def remove_non_round(segmentation, props, ff_cutoff):
    for i, p in zip(segmentation.identifiers, props):
        region = segmentation.region_by_identifier(i)
#       print(form_factor(p))
        if form_factor(p) < ff_cutoff:
            segmentation[region] = 0
    return segmentation


@transformation
def fill_holes(image, min_size):
    tmp_autowrite_on = AutoWrite.on
    AutoWrite.on = False
    image = invert(image)
    image = remove_small_objects(image, min_size=min_size)
    image = invert(image)
    AutoWrite.on = tmp_autowrite_on
    return image


@transformation
def distance(image):
    """Return result of an exact euclidean distance transform."""
    return scipy.ndimage.distance_transform_edt(image)


@transformation
def local_maxima(image, footprint=None, labels=None):
    """Return local maxima."""
    return skimage.feature.peak_local_max(image,
                                          indices=False,
                                          footprint=footprint,
                                          labels=labels)


def find_tubes(input_file, output_dir=None):
    bname = os.path.basename(input_file)
    name, suffix = bname.split(".")
    name = "tubes-" + name + ".png"
    if output_dir:
        name = os.path.join(output_dir, name)

    image = Image.from_file(input_file)
    intensity = mean_intensity_projection(image)
    image = find_edges_sobel(intensity)
    image = remove_scalebar(image, 0)
    image = threshold_otsu(image)
    image = dilate_binary(image, selem=disk(3))
    image = erode_binary(image, selem=disk(3))
    image = remove_small_objects(image, min_size=500)
    image = fill_holes(image, min_size=500)
    image = erode_binary(image, selem=disk(3))
    image = remove_small_objects(image, min_size=200)

    segmentation = connected_components(image, background=0)

    ann = AnnotatedImage.from_grayscale(intensity)
    for i in segmentation.identifiers:
        region = segmentation.region_by_identifier(i)
        ann.mask_region(region.border.dilate(), color=pretty_color(i))

    with open(name, "wb") as fh:
        fh.write(ann.png())

    return segmentation


def find_grains(input_file, output_dir=None):
    bname = os.path.basename(input_file)
    name, suffix = bname.split(".")
    name = "grains-" + name + ".png"
    if output_dir:
        name = os.path.join(output_dir, name)

    image = Image.from_file(input_file)
    intensity = mean_intensity_projection(image)
    image = remove_scalebar(intensity, np.mean(intensity))
    image = threshold_abs(image, 75)
    image = invert(image)
    image = fill_holes(image, min_size=500)
    image = erode_binary(image, selem=disk(4))
    image = remove_small_objects(image, min_size=500)
    image = dilate_binary(image, selem=disk(4))

    dist = distance(image)
    seeds = local_maxima(dist)
    seeds = dilate_binary(seeds)  # Merge spurious double peaks.
    seeds = connected_components(seeds, background=0)

    segmentation = watershed_with_seeds(dist, seeds=seeds, mask=image)
    initial_segmentation = np.copy(segmentation)
    print "initial", np.min(initial_segmentation), np.max(initial_segmentation), initial_segmentation.dtype, np.sum(initial_segmentation), type(initial_segmentation)

    # Remove spurious blobs.
    segmentation = remove_large_segments(segmentation, max_size=3000)
    segmentation = remove_small_segments(segmentation, min_size=500)
    props = skimage.measure.regionprops(segmentation)
    segmentation = remove_non_round(segmentation, props, 0.6)

    difficult = initial_segmentation - segmentation
    print "difficult", np.min(difficult), np.max(difficult), difficult.dtype, np.sum(difficult), type(difficult)
    print "initial", np.min(initial_segmentation), np.max(initial_segmentation), initial_segmentation.dtype, np.sum(initial_segmentation), type(initial_segmentation)
    print "segment", np.min(segmentation), np.max(segmentation), segmentation.dtype, np.sum(segmentation), type(segmentation)

    ann = AnnotatedImage.from_grayscale(intensity)
    for i in segmentation.identifiers:
        region = segmentation.region_by_identifier(i)
        ann.mask_region(region.border.dilate(), color=pretty_color(i))

    for i in difficult.identifiers:
        print("difficult", i)
        region = difficult.region_by_identifier(i) 
        ann.mask_region(region.border.dilate(4), color=pretty_color(i))


    with open(name, "wb") as fh:
        fh.write(ann.png())

    return segmentation

    

def annotate(input_file, output_dir=None):
#   tubes = find_tubes(input_file, output_dir)
    grains = find_grains(input_file, output_dir)


def analyse_all(input_dir, output_dir):
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith("jpg"):
            continue
        if fname.lower().startswith("leicalogo"):
            continue
        fpath = os.path.join(input_dir, fname)
        print(fpath)
        annotate(fpath, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="Input jpg file")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if os.path.isfile(args.input_file):
        annotate(args.input_file)
    elif os.path.isdir(args.input_file):
        AutoWrite.on = False
        if args.output_dir is None:
            parser.error("Need to specify --output-dir option when using an input dir")
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        analyse_all(args.input_file, args.output_dir)
    else:
        parser.error("{} not a file or directory".format(args.input_file))

if __name__ == "__main__":
    main()

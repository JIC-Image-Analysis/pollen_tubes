"""Segment pollen tubes."""

import os
import argparse
import warnings
import math
import logging

import numpy as np
import scipy.ndimage
from skimage.morphology import disk
import skimage.feature

from jicbioimage.core.image import Image
from jicbioimage.core.transform import transformation
from jicbioimage.core.util.color import pretty_color
from jicbioimage.core.util.array import normalise
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


__version__ = "0.3.0"

# Suppress spurious scikit-image warnings.
#warnings.filterwarnings("ignore", module="skimage.exposure._adapthist")
#warnings.filterwarnings("ignore", module="skimage.util.dtype")
warnings.filterwarnings("ignore", module="skimage.io._io")

# Setup logging with a stream handler.
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

AutoName.prefix_format = "{:03d}_"


def form_factor(prop):
    """Return form factor circularity measure."""
    return (4 * math.pi * prop.area) / float(prop.perimeter)**2


def centroid(region):
    """Return y, x centroid coordinates."""
    return tuple([int(np.mean(ia)) for ia in region.index_arrays])


@transformation
def remove_scalebar(image, value):
    """Remove the scale bar from the image."""
    image[-42:, -154:] = value
    return image


@transformation
def threshold_abs(image, threshold):
    return image > threshold


@transformation
def remove_large_segments(segmentation, max_size):
    """Remove large regions from a segmentation."""
    for i in segmentation.identifiers:
        region = segmentation.region_by_identifier(i)
        if region.area > max_size:
            segmentation[region] = 0
    return segmentation


@transformation
def remove_small_segments(segmentation, min_size):
    """Remove small regions from a segmentation."""
    for i in segmentation.identifiers:
        region = segmentation.region_by_identifier(i)
        if region.area < min_size:
            segmentation[region] = 0
    return segmentation


@transformation
def remove_non_round(segmentation, props, ff_cutoff):
    """Remove non-round regions from a segmentation."""
    for i, p in zip(segmentation.identifiers, props):
        region = segmentation.region_by_identifier(i)
#       print(form_factor(p))
        if form_factor(p) < ff_cutoff:
            segmentation[region] = 0
    return segmentation


@transformation
def fill_holes(image, min_size):
    """Return image with holes filled in."""
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
    """Return pollen tube segmentation."""
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

    return segmentation


def find_grains(input_file, output_dir=None):
    """Return tuple of segmentaitons (grains, difficult_regions)."""
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
    # Need a copy to avoid over-writing original.
    initial_segmentation = np.copy(segmentation)

    # Remove spurious blobs.
    segmentation = remove_large_segments(segmentation, max_size=3000)
    segmentation = remove_small_segments(segmentation, min_size=500)
    props = skimage.measure.regionprops(segmentation)
    segmentation = remove_non_round(segmentation, props, 0.6)

    difficult = initial_segmentation - segmentation

    return segmentation, difficult


@transformation
def remove_tubes_not_touching_grains(tubes, grains):
    """Return tube segments that overlap with grain segemnts."""
    for i in tubes.identifiers:
        region = tubes.region_by_identifier(i)
        overlap = np.sum(grains[region.dilate()])
        if overlap == 0:
            tubes[region] = 0
    return tubes


@transformation
def remove_tubes_that_are_grains(tubes, grains):
    """Return tube segments that do not engulf grain segments."""
    grains = grains.astype(bool)
    for i in tubes.identifiers:
        region = tubes.region_by_identifier(i)
        shared = np.sum(np.logical_and(grains, region))
        tube_only = np.sum(np.logical_and(np.logical_not(grains), region))
        overlap_ratio = float(shared) / (tube_only + shared)
        if overlap_ratio > 0.5:
            tubes[region] = 0
    return tubes


def annotate(input_file, output_dir=None):
    """Write an annotated image to disk."""
    logger.info("---")
    logger.info('Input image: "{}"'.format(os.path.abspath(input_file)))
    image = Image.from_file(input_file)
    intensity = mean_intensity_projection(image)
    norm_intensity = normalise(intensity)
    norm_rgb = np.dstack([norm_intensity, norm_intensity, norm_intensity])

    bname = os.path.basename(input_file)
    name, suffix = bname.split(".")
    png_name = name + ".png"
    csv_name = name + ".csv"
    if output_dir:
        png_name = os.path.join(output_dir, png_name)
        csv_name = os.path.join(output_dir, csv_name)

    tubes = find_tubes(input_file, output_dir)
    grains, difficult = find_grains(input_file, output_dir)
    tubes = remove_tubes_not_touching_grains(tubes, grains)
    tubes = remove_tubes_that_are_grains(tubes, grains)

    ann = AnnotatedImage.from_grayscale(intensity)

    num_grains = 0
    for n, i in enumerate(grains.identifiers):
        n = n + 1
        region = grains.region_by_identifier(i)
        ann.mask_region(region.inner.inner.inner.border.dilate(),
                        color=(0, 255, 0))
        num_grains = n

    num_tubes = 0
    for n, i in enumerate(tubes.identifiers):
        n = n + 1
        region = tubes.region_by_identifier(i)
        highlight = norm_rgb * pretty_color(i)
        ann[region] = highlight[region]
        ann.mask_region(region.dilate(3).border.dilate(3),
                        color=pretty_color(i))
        num_tubes = n

    ann.text_at("Num grains: {:3d}".format(num_grains), 10, 10, antialias=True,
                color=(0, 255, 0), size=48)
    logger.info("Num grains: {:3d}".format(num_grains))

    ann.text_at("Num tubes : {:3d}".format(num_tubes), 10, 60, antialias=True,
                color=(255, 0, 255), size=48)
    logger.info("Num tubes : {:3d}".format(num_tubes))

    logger.info('Output image: "{}"'.format(os.path.abspath(png_name)))
    with open(png_name, "wb") as fh:
        fh.write(ann.png())

    logger.info('Output csv: "{}"'.format(os.path.abspath(csv_name)))
    with open(csv_name, "w") as fh:
        fh.write("img,grains,tubes\n")
        fh.write("{},{},{}\n".format(png_name, num_grains, num_tubes))

    return png_name, num_grains, num_tubes


def analyse_all(input_dir, output_dir):
    summary_name = os.path.basename(input_dir.rstrip("/"))
    summary_name = "summary_" + summary_name + ".csv"
    summary_name = os.path.join(output_dir, summary_name)
    logger.info('Summary csv: "{}"'.format(os.path.abspath(summary_name)))
    with open(summary_name, "w") as fh:
        fh.write("img,grains,tubes\n")
        for fname in os.listdir(input_dir):
            if not fname.lower().endswith("jpg"):
                continue
            if fname.lower().startswith("leicalogo"):
                continue
            fpath = os.path.join(input_dir, fname)
            png_name, num_grains, num_tubes = annotate(fpath, output_dir)
            fh.write("{},{},{}\n".format(png_name, num_grains, num_tubes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input file/directory")
    parser.add_argument("output_dir", help="Output directory")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    AutoName.directory = args.output_dir

    # Create file handle logger.
    fh = logging.FileHandler(os.path.join(args.output_dir, "log"), mode="w")
    fh.setLevel(logging.DEBUG)
    format_ = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info("Script name: {}".format(__file__))
    logger.info("Script version: {}".format(__version__))
    if os.path.isfile(args.input):
        annotate(args.input, args.output_dir)
    elif os.path.isdir(args.input):
        AutoWrite.on = False
        analyse_all(args.input, args.output_dir)
    else:
        parser.error("{} not a file or directory".format(args.input))

if __name__ == "__main__":
    main()

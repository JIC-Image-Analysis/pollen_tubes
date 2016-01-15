"""Segment pollen tubes."""

import os
import argparse
import warnings

import numpy as np
from skimage.morphology import disk

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
from jicbioimage.segment import connected_components
from jicbioimage.illustrate import AnnotatedImage


AutoName.prefix_format = "{:03d}_"


@transformation
def remove_scalebar(image, value):
    """Remove the scale bar from the image."""
    image[-42:,-154:] = value
    return image

@transformation
def fill_holes(image, min_size):
    tmp_autowrite_on = AutoWrite.on
    AutoWrite.on = False
    image = invert(image)
    image = remove_small_objects(image, min_size=min_size)
    image = invert(image)
    AutoWrite.on = tmp_autowrite_on
    return image

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
    image = threshold_otsu(image)
    image = invert(image)
    image = erode_binary(image, selem=disk(4))
    image = remove_small_objects(image, min_size=500)
    image = fill_holes(image, min_size=500)
    image = dilate_binary(image, selem=disk(4))

    segmentation = connected_components(image, background=0)

    ann = AnnotatedImage.from_grayscale(intensity)
    for i in segmentation.identifiers:
        region = segmentation.region_by_identifier(i)
        ann.mask_region(region.border.dilate(), color=pretty_color(i))

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

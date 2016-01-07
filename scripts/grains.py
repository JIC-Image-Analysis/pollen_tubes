"""Identify individual pollen grains."""

import os
import argparse
import warnings
import math

import numpy as np

import skimage.measure

from jicbioimage.core.image import Image
from jicbioimage.core.io import AutoWrite
from jicbioimage.core.region import Region
from jicbioimage.core.transform import transformation

from jicbioimage.core.util.color import pretty_color
from jicbioimage.transform import (
    mean_intensity_projection,
    remove_small_objects,
    invert,
)
from jicbioimage.segment import connected_components
from jicbioimage.illustrate import AnnotatedImage

# Suppress spurious scikit-image warnings.
warnings.filterwarnings("ignore", module="skimage.exposure._adapthist")
warnings.filterwarnings("ignore", module="skimage.util.dtype")
warnings.filterwarnings("ignore", module="skimage.io._io")

def form_factor(prop):
    return (4 * math.pi * prop.area) / float(prop.perimeter)**2


@transformation
def threshold_abs(image, threshold):
    return image > threshold


@transformation
def remove_non_round(segmentation, props, ff_cutoff):
    for i, p in zip(segmentation.identifiers, props):
        region = segmentation.region_by_identifier(i)
        if form_factor(p) < ff_cutoff:
            segmentation[region] = 0
    return segmentation
    
    
    


def grains(input_file, threshold, output_dir=None):
    bname = os.path.basename(input_file)
    name, suffix = bname.split(".")
    name = name + ".png"
    if output_dir:
        name = os.path.join(output_dir, name)

    image = Image.from_file(input_file)
    intensity = mean_intensity_projection(image)

    mask = threshold_abs(intensity, threshold)
    mask = remove_small_objects(mask, min_size=100)
    mask = invert(mask)
    mask = remove_small_objects(mask, min_size=500)

    segmentation = connected_components(mask, background=0)
    props = skimage.measure.regionprops(segmentation)
    segmentation = remove_non_round(segmentation, props, 0.7)

    mask[np.where(segmentation != 0)] = 0
    mask_region = Region(mask)

    ann = AnnotatedImage.from_grayscale(intensity)
    for i in segmentation.identifiers:
        region = segmentation.region_by_identifier(i)
#       ann.mask_region(region.border.dilate(), color=pretty_color(i))
        ann.mask_region(region.border.dilate(), color=(0, 255, 0))
        ann.mask_region(mask_region.border.dilate(), color=(255, 0, 255))

    with open(name, "wb") as fh:
        fh.write(ann.png())

def analyse_all(input_dir, threshold, output_dir):
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith("jpg"):
            continue
        if fname.lower().startswith("leicalogo"):
            continue
        fpath = os.path.join(input_dir, fname)
        print(fpath)
        grains(fpath, threshold, output_dir)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="Input jpg file")
    parser.add_argument("-t", "--threshold", type=int, default=75)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if os.path.isfile(args.input_file):
        grains(args.input_file, args.threshold)
    elif os.path.isdir(args.input_file):
        AutoWrite.on = False
        if args.output_dir is None:
            parser.error("Need to specify --output-dir option when using an input dir")
        if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        analyse_all(args.input_file, args.threshold, args.output_dir)
    else:
        parser.error("{} not a file or directory".format(args.input_file))


if __name__ == "__main__":
    main()


"""Segment pollen tubes."""

import os
import argparse
import warnings

from jicbioimage.core.image import Image
from jicbioimage.core.io import AutoWrite
from jicbioimage.transform import (
    mean_intensity_projection,
)

def tubes(input_file, output_dir=None):
    bname = os.path.basename(input_file)
    name, suffix = bname.split(".")
    name = name + ".png"
    if output_dir:
        name = os.path.join(output_dir, name)

    image = Image.from_file(input_file)
    intensity = mean_intensity_projection(image)


def analyse_all(input_dir, threshold, output_dir):
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith("jpg"):
            continue
        if fname.lower().startswith("leicalogo"):
            continue
        fpath = os.path.join(input_dir, fname)
        print(fpath)
        tubes(fpath, output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", help="Input jpg file")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    if os.path.isfile(args.input_file):
        tubes(args.input_file)
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

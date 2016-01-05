"""Analyse all JPEG images in a directory."""

import os
import os.path
import argparse

from segment import analyse

from jicbioimage.core.io import AutoWrite
AutoWrite.on = False

def analyse_all(input_dir, output_dir):
    for fname in os.listdir(input_dir):
        if not fname.lower().endswith("jpg"):
            continue
        if fname.lower().startswith("leicalogo"):
            continue
        fpath = os.path.join(input_dir, fname)
        print(fpath)
        analyse(fpath, output_dir)
    

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", help="Input jpg dir")
    parser.add_argument("output_dir", help="Output dir")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        parser.error("No such directory: {}".format(args.input_file))

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)

    analyse_all(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

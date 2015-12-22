#!/usr/bin/bash

mkdir -p data/crop

convert -crop 2100x2500+7300+2400 data/raw/TileScan_001_Merging001_ch00.jpg data/crop/TileScan_001_Merging001_ch00.jpg
convert -crop 1000x1000+8300+3400 data/raw/TileScan_001_Merging001_ch00.jpg data/crop/small.jpg
convert -crop 1000x1000+8300+3400 data/raw/TileScan_001_Merging001_ch00.jpg data/crop/small.tiff

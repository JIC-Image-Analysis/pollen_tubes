# README

Count pollen grains and tubes.

```
$ python scripts/annotate.py data/raw/wt/TileScan_001/ -d output_dir
```

## Running the code

To run the code, first create the data and output directories:

```
mkdir {data,output}
```

Then copy the data to be analysed into the data directory. Then, start the
docker container with:

```
bash runcontainer.sh
```

If the data has been collected using a Leica microscope run:

```
python /code/annotate.py /data/ /output/
```

If the data has been collected using a Nikon E800 microscope run:

```
python /code/nikonE800_annotate.py /data/ /output/
```

The analysis will be available in the output directory.

# Methodology

## Image analysis

### Summary

In order to aid in the counting of the number of germinated and ungerminated
pollen grains two scripts were written in Python.

The first script identified and counted the number of pollen grains in images
captured using the Nikon E800 microscope.

The second script identified and counted both the number of pollen grains and
the number of pollen tubes in images captured using the Leica DM6000 microscope.
By subtracting the number of pollen tubes from the number of pollen grains the
number of ungerminated grains could be identified.

Both scripts produced annotated images that were manually inspected to verify
that the algorithm had worked as intended.

### Details

The script to analyse the Nikon E800 images included a number of steps to
generate an image suitable for watershedding:

1. A median threshold
2. Inversion
3. Erosion
4. Dilation
5. Removal of small objects
6. Hole filling

Seeds for the watershed algorithm were obtained by applying a distance
transform to the binary image followed by the identification of local maxima.

The grains were then identified by applying the watershed algorithm using
the distance image as input, with the seeds from the local maxima and the
binary image as a mask.

The script to analyse the Leica DM6000 images was more involved. It took
advantage of the high resolution of the images.

First of all it identified the grains in a similar manner to that outlined
above. This was followed by the removal of spurious blobs based on the area and
circularity.

Pollen tubes were then identified by virtue of the high resolution of the data,
in particular the fact that the tubes were characterised by two parallel lines.
By finding edges using a Sobel tansform the two parallel lines were converted to
four parallel lines. Using dilation and erosion the outlines of pollen grains
were removed whilst the pollen tube region remained intact.

The exact implementation of the two algorithms are available in the source code
hosted on GitHub available under the open source MIT licence
https://github.com/JIC-Image-Analysis/pollen_tubes .

The image analysis pipeline was written in Python and made use of the [JIC
BioImage framework](https://github.com/JIC-CSB/jicbioimage).  The image
anslysis made use of several scientific Python packages including numpy [ref1],
scipy [ref2], and scikit-image [ref3].

[ref1] http://ieeexplore.ieee.org/xpl/articleDetails.jsp?arnumber=5725236
[ref2] Jones E, Oliphant E, Peterson P, et al. SciPy: Open Source Scientific Tools for Python, 2001-, http://www.scipy.org/ [Online; accessed 2016-04-11].
[ref3] https://peerj.com/articles/453/

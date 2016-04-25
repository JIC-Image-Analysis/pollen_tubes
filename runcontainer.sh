CONTAINER=jicscicomp/jicbioimage
docker run -it --rm -v `pwd`/data:/data -v `pwd`/scripts:/code -v `pwd`/output:/output $CONTAINER


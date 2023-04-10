#!/bin/bash

VIVADO_INSTALL_DIR=/opt/Xilinx
VIVADO_VERSION=2020.1

# If running in docker image we would first need to activate the proper conda environment
#. activate hls4ml-py36

# Convert models in keras-models.txt
./convert-keras-models.sh -x -f keras-models.txt

# Alternatively, keras-to-hls script can be called, with the model name(s) specified, i.e.:
#./keras-to-hls.sh KERAS_1layer KERAS_conv1d_small
./keras-to-hls.sh -b alveo-u250 -B VivadoAccelerator -x xcu250-figd2104-2L-e KERAS_3layer
./keras-to-hls.sh -b pynq-z2 -B VivadoAccelerator -x xc7z020clg400-1 KERAS_3layer
# KERAS_3layer b:pynq-z2 B:VivadoAccelerator x:xc7z020clg400-1 s:Resource

# Build the projects generated by keras-to-hls script.
# Remove parameter -s to disable synthesis. -p controls the number of parallel tasks
./build-prj.sh -i ${VIVADO_INSTALL_DIR} -v ${VIVADO_VERSION} -c -s -p 2

# Go through the generated reports and print out basic information.
# Reports are available if synthesis is enabled.
./gather-reports.sh -b

# Clean-up at the end
#./cleanup.sh

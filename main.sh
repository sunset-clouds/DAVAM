#!/bin/bash

# configure pip to use internal source
unset http_proxy
unset https_proxy
mkdir -p ~/.pip/ \
    && echo "[global]"                                              > ~/.pip/pip.conf \
    && echo "index-url = http://mirror-sng.oa.com/pypi/web/simple/" >> ~/.pip/pip.conf \
    && echo "trusted-host = mirror-sng.oa.com"                      >> ~/.pip/pip.conf
cat ~/.pip/pip.conf

# install Python packages with Internet access

# add the current directory to PYTHONPATH
export PYTHONPATH=${PYTHONPATH}:`pwd`
export LD_LIBRARY_PATH=/opt/ml/disk/local/cuda/lib64:$LD_LIBRARY_PATH

SAVER_DIR=/opt/ml/disk/discrete_vae/saver
mkdir -p ${SAVER_DIR}

# execute the main script
# mkdir models
EXTRA_ARGS=`cat ./extra_args`
python main.py --saver_dir ${SAVER_DIR} ${EXTRA_ARGS}

# remove *.pyc files
find . -name "*.pyc" -exec rm -f {} \;

#!/bin/bash
cd ./models/networks/block_extractor
python setup.py clean --all install --user

cd ..
cd local_attn_reshape
python setup.py clean --all install --user

cd ..
cd resample2d_package
python setup.py clean --all install --user

cd ..
cd correlation_package
python setup.py clean --all install --user
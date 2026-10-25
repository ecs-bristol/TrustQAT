#!/bin/sh
# Download the HLS4ML LHC jet dataset (150 particles, Zenodo 3602260) and build
# dataset/150c-train.h5 and dataset/150c-test.h5 in the current directory.
# Adapted from JEDI-linear (Apache-2.0); see NOTICE.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"

curl -L -o hls4ml_LHCjet_150p_train.tar.gz "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_train.tar.gz?download=1"
curl -L -o hls4ml_LHCjet_150p_val.tar.gz "https://zenodo.org/records/3602260/files/hls4ml_LHCjet_150p_val.tar.gz?download=1"

tar -xzf hls4ml_LHCjet_150p_train.tar.gz
tar -xzf hls4ml_LHCjet_150p_val.tar.gz

mkdir -p dataset
python3 "$HERE/prepare_dataset.py" -i ./train/ -o dataset/150c-train.h5 -j 4
python3 "$HERE/prepare_dataset.py" -i ./val/ -o dataset/150c-test.h5 -j 4

rm -rf ./train/ ./val/

#!/usr/bin/env bash
apt install unzip
pip install -r requirements.txt
wget https://github.com/google-research/distilling-step-by-step/raw/main/datasets.zip
unzip datasets.zip
rm -f datasets.zip

#!/bin/bash

pip3 install -r requirements.txt
unzip good_instances.zip
mkdir solutions
python3 experiments.py

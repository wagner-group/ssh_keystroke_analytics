#!/bin/bash

pip install -r requirements.txt
python setup.py install
cd examples/data
bash gen.sh
cd ../../

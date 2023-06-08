#!/bin/bash

mkdir train
cd train
python ../generate.py 
cd ..
cp -r train eval
cp -r train testing


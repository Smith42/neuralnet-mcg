#!/bin/bash

for i in {0..2}
do
  echo $i
  python3 ./diagAUC.py $i
done

#!/bin/bash

for i in {0..4}
do
  echo $i
  python3 ./600plus_eachEpoch.py $i
done

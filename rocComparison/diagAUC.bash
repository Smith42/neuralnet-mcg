#!/bin/bash

for i in {0..3}
do 
  for j in {0..9}
  do
    echo $i $j
    python3 ./diagAUC.py $i $j
  done
done

#!/bin/bash

for i in {0..2}
do 
  for j in {0..8}
  do
    echo $i $j
    python3 ./diagAUC.py $i $j
  done
done

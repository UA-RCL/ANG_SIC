#!/bin/bash

declare -a filenames=("training.bin" "testing.bin")
declare -a fileIDs=("1cI81LwJiFNu1ydnrVikSKbBIzxTPcnLn" "1mk7wxm5iyWM7_NMyldVodp5i2LkcPWef")

i=0
for file in "${filenames[@]}"; do
    echo "______________________________________"
    echo "Downloading MNIST dataset- ${file}"
    echo "______________________________________"
    echo "curl -L \"https://drive.usercontent.google.com/download?id=${fileIDs[$i]}&confirm=xxx\" -o ${file}"
    curl -L "https://drive.usercontent.google.com/download?id=${fileIDs[$i]}&confirm=xxx" -o ${file}
    ((i=i+1))
    echo "______________________________________"
done

#!/bin/bash

OUTPUT="results.txt"
echo "Forest Fire Benchmark Results" > $OUTPUT
echo "==============================" >> $OUTPUT

for N in 100 400 800 1000 1200; do
    echo "" >> $OUTPUT
    echo "=== Grid Size: ${N}x${N} ===" >> $OUTPUT
    
    # Change N in main.cpp, recompile, run both devices
    sed -i "s/const int N\s*=\s*[0-9]*/const int N               = $N/" src/main.cpp
    g++ src/main.cpp -o fire.exe \
      -I"/d/msys64/ucrt64/include" \
      -I"include" \
      -L"/d/msys64/ucrt64/lib" \
      -lOpenCL -lgdi32 -luser32 -lkernel32

    echo "-- GPU --" >> $OUTPUT
    ./fire.exe gpu >> $OUTPUT

    echo "-- CPU --" >> $OUTPUT
    ./fire.exe cpu >> $OUTPUT
done

echo "" >> $OUTPUT
echo "Benchmark complete." >> $OUTPUT
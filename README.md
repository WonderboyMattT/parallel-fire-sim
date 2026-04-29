# Parallel Forest Fire Simulation

CMP3752 Parallel Programming — Assessment 1

A parallel cellular automaton simulation of forest fire spread, implemented in C++ using OpenCL. Supports execution on both CPU and GPU devices.

## Project Structure

Parallel-Fire/
├─ .vscode/
│ ├─ c_cpp_properties.json
│ ├─ launch.json
│ └─ settings.json
├─ include/
│ └─ CImg.h
├─ kernels/
│ └─ fire.cl
├─ src/
│ └─ main.cpp
├─ .gitignore
├─ fire.exe
├─ README.md
└─ results.txt

## Dependencies

- OpenCL (via MSYS2 UCRT64: `mingw-w64-ucrt-x86_64-opencl-icd`)
- CImg (included in `include/`)
- g++ via MSYS2 UCRT64

## Compile

Run from the project root in a MSYS2 UCRT64 terminal:

```bash
g++ src/main.cpp -o fire.exe \
  -I"/d/msys64/ucrt64/include" \
  -I"include" \
  -L"/d/msys64/ucrt64/lib" \
  -lOpenCL -lgdi32 -luser32 -lkernel32
```

## Run

### Benchmark mode (no visualisation)

```bash
./fire.exe gpu     # run on GPU
./fire.exe cpu     # run on CPU
```

### Visualisation mode

```bash
./fire.exe gpu visualise
./fire.exe cpu visualise
```

> Note: run the exe by double-clicking from File Explorer if the display doesn't open from the terminal.

## Simulation Parameters

Set in `src/main.cpp`:

| Parameter       | Value    | Description                              |
| --------------- | -------- | ---------------------------------------- |
| `probTree`      | 0.8      | Probability a cell contains a tree       |
| `probBurning`   | 0.01     | Probability an initial tree is burning   |
| `probImmune`    | 0.3      | Probability a tree is immune per step    |
| `probLightning` | 0.001    | Probability of lightning strike per step |
| `N`             | 100–1200 | Grid size (N×N)                          |
| `STEPS`         | 50       | Number of simulation steps               |

Alter cimg::wait() to increase step time for slower visualisation (1000 = 1 second per step)
cellsize for larger "pixels"
N for grid size

## Benchmarking

To run the full benchmark across all required grid sizes on both devices:

```bash
bash ./benchmark.sh
```

Results are saved to `results.txt`.

## Cell States

| Value | State   | Colour     |
| ----- | ------- | ---------- |
| 0     | Empty   | Brown      |
| 1     | Tree    | Green      |
| 2     | Burning | Orange-red |

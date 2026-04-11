// main.cpp

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include "CImg.h"
using namespace cimg_library;

std::string loadKernelSource(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open kernel file: " + path);
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
}

// ─────────────────────────────────────────────
// Helper: convert int grid to RGB CImg
// 0 = empty  -> brown  (80, 50, 20)
// 1 = tree   -> green       (34, 139, 34)
// 2 = burning -> orange (220, 60, 10)
// ─────────────────────────────────────────────
CImg<unsigned char> gridToImage(const std::vector<int>& grid, int N, int cellSize = 4) {
    CImg<unsigned char> img(N * cellSize, N * cellSize, 1, 3, 0);
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            int val = grid[y * N + x];
            unsigned char r, g, b;
            if      (val == 0) { r = 80;  g = 50;  b = 20;  }  // empty
            else if (val == 1) { r = 34;  g = 139; b = 34;  }  // tree
            else               { r = 220; g = 60;  b = 10;  }  // burning
            // Fill cellSize x cellSize block
            for (int py = 0; py < cellSize; py++)
                for (int px = 0; px < cellSize; px++) {
                    img(x * cellSize + px, y * cellSize + py, 0, 0) = r;
                    img(x * cellSize + px, y * cellSize + py, 0, 1) = g;
                    img(x * cellSize + px, y * cellSize + py, 0, 2) = b;
                }
        }
    }
    return img;
}

int main(int argc, char* argv[]) {
    try {

        // ─────────────────────────────────────────────
        // COMMAND LINE ARGUMENTS
        // Usage: ./fire.exe [cpu|gpu] [visualise]
        // e.g:   ./fire.exe gpu
        //        ./fire.exe cpu visualise
        // Defaults to GPU, no visualisation
        // ─────────────────────────────────────────────
        bool useCPU    = false;
        bool visualise = false;

        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "cpu")       useCPU    = true;
            if (arg == "gpu")       useCPU    = false;
            if (arg == "visualise") visualise = true;
        }

        // ─────────────────────────────────────────────
        // SIMULATION PARAMETERS
        // ─────────────────────────────────────────────
        const int N               = 100; // Grid size (N x N)
        const int STEPS           = 50; // Number of simulation steps
        const float probTree      = 0.8f; // Initial tree density
        const float probBurning   = 0.01f; // Initial burning tree density
        const float probImmune    = 0.3f; // Probability a tree is immune to fire (won't burn)
        const float probLightning = 0.001f; // Probability of lightning strike
        const int gridSize        = N * N; // Total number of cells

        // ─────────────────────────────────────────────
        // PLATFORM & DEVICE SELECTION
        // ─────────────────────────────────────────────
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) { std::cerr << "No OpenCL platforms found.\n"; return 1; }

        cl::Device cpuDevice, gpuDevice;
        bool foundCPU = false, foundGPU = false;

        for (auto& platform : platforms) {
            std::vector<cl::Device> devices;
            platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
            for (auto& device : devices) {
                cl_device_type type = device.getInfo<CL_DEVICE_TYPE>();
                if (type == CL_DEVICE_TYPE_CPU && !foundCPU) { cpuDevice = device; foundCPU = true; }
                if (type == CL_DEVICE_TYPE_GPU && !foundGPU) { gpuDevice = device; foundGPU = true; }
            }
        }

        if (!foundCPU && useCPU) { std::cerr << "No CPU device found.\n"; return 1; }
        if (!foundGPU && !useCPU) { std::cerr << "No GPU device found.\n"; return 1; }

        cl::Device activeDevice = useCPU ? cpuDevice : gpuDevice;

        std::cout << "====================================\n";
        std::cout << "Device: " << activeDevice.getInfo<CL_DEVICE_NAME>() << "\n";
        std::cout << "Grid:   " << N << " x " << N << "\n";
        std::cout << "Steps:  " << STEPS << "\n";
        std::cout << "Mode:   " << (visualise ? "visualise" : "benchmark") << "\n";
        std::cout << "====================================\n";


        // ─────────────────────────────────────────────
        // CONTEXT, QUEUE, PROGRAM
        // ─────────────────────────────────────────────
        cl::Context context(activeDevice);
        cl::CommandQueue queue(context, activeDevice, CL_QUEUE_PROFILING_ENABLE);

        std::string kernelSrc = loadKernelSource("kernels/fire.cl");
        cl::Program program(context, kernelSrc);
        try {
            program.build({ activeDevice });
        } catch (const cl::Error& e) {
            std::cerr << "Kernel build failed:\n"
                      << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(activeDevice) << "\n";
            return 1;
        }

        // ─────────────────────────────────────────────
        // DOUBLE BUFFERING
        // Two buffers: read from one, write to other,
        // then swap each step to avoid race conditions.
        // ─────────────────────────────────────────────
        cl::Buffer bufferA(context, CL_MEM_READ_WRITE, sizeof(int) * gridSize);
        cl::Buffer bufferB(context, CL_MEM_READ_WRITE, sizeof(int) * gridSize);

        // ─────────────────────────────────────────────
        // TOTAL WALL CLOCK START
        // Measures everything including transfers
        // ─────────────────────────────────────────────
        auto wallStart = std::chrono::steady_clock::now();

        // ─────────────────────────────────────────────
        // INIT GRID KERNEL
        // ─────────────────────────────────────────────
        cl::Kernel initKernel(program, "init_grid");
        cl_ulong seed = (cl_ulong)std::chrono::steady_clock::now().time_since_epoch().count();

        initKernel.setArg(0, bufferA);
        initKernel.setArg(1, N);
        initKernel.setArg(2, probTree);
        initKernel.setArg(3, probBurning);
        initKernel.setArg(4, seed);

        cl::Event initEvent;
        queue.enqueueNDRangeKernel(initKernel, cl::NullRange,
                                   cl::NDRange(gridSize), cl::NullRange,
                                   nullptr, &initEvent);
        queue.finish();

        // OpenCL event profiling for init kernel
        double initKernelTime = (initEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                                 initEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-6;


        /* ─────────────────────────────────────────────
        // PRINT INITIAL STATE
        // ─────────────────────────────────────────────
        std::vector<int> hostGrid(gridSize);
        queue.enqueueReadBuffer(bufferA, CL_TRUE, 0, sizeof(int) * gridSize, hostGrid.data());

        std::cout << "\nInitial grid top-left 10x10:\n";
        for (int y = 0; y < 10; y++) {
            for (int x = 0; x < 10; x++)
                std::cout << hostGrid[y * N + x] << " ";
            std::cout << "\n";
        }

        auto countCells = [&](const std::string& label) {
            int empty = 0, trees = 0, burning = 0;
            for (int v : hostGrid) {
                if (v == 0) empty++;
                else if (v == 1) trees++;
                else if (v == 2) burning++;
            }
            std::cout << label << " -- Empty: " << empty
                      << "  Trees: " << trees
                      << "  Burning: " << burning << "\n";
        };
        countCells("Initial counts");

        */

        // ─────────────────────────────────────────────
        // VISUALISATION SETUP
        // cellSize scales each grid cell up so it's
        // visible — 4px per cell on a 100x100 grid
        // gives a 400x400 window.
        // ─────────────────────────────────────────────
        //const int cellSize = 4; // Scale factor for display (Smaller for larger grids, but bigger is slower to render)
        //CImg<unsigned char> img = gridToImage(hostGrid, N, cellSize);
        //CImgDisplay display(img, "Forest Fire Simulation");

        std::vector<int> hostGrid(gridSize);
        std::unique_ptr<CImgDisplay> display;

        if (visualise) {
            queue.enqueueReadBuffer(bufferA, CL_TRUE, 0,
                                    sizeof(int) * gridSize, hostGrid.data());
            const int cellSize = (N <= 100) ? 6 : (N <= 400) ? 2 : 1;
            CImg<unsigned char> img = gridToImage(hostGrid, N, cellSize);
            display = std::make_unique<CImgDisplay>(img, "Forest Fire Simulation");
        }

        // ─────────────────────────────────────────────
        // FIRE SPREAD SIMULATION LOOP
        // ─────────────────────────────────────────────
        cl::Kernel spreadKernel(program, "fire_spread");
        cl::Buffer* bufIn  = &bufferA;
        cl::Buffer* bufOut = &bufferB;

        double totalSpreadTime = 0.0;

        for (int step = 0; step < STEPS; step++) {

            if (visualise && display && display->is_closed()) break;

            cl_ulong stepSeed = seed + (cl_ulong)step * 999983UL;

            spreadKernel.setArg(0, *bufIn);
            spreadKernel.setArg(1, *bufOut);
            spreadKernel.setArg(2, N);
            spreadKernel.setArg(3, probImmune);
            spreadKernel.setArg(4, probLightning);
            spreadKernel.setArg(5, stepSeed);

            cl::Event spreadEvent;
            queue.enqueueNDRangeKernel(spreadKernel, cl::NullRange,
                                       cl::NDRange(gridSize), cl::NullRange,
                                       nullptr, &spreadEvent);
            queue.finish();

            totalSpreadTime += (spreadEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                                spreadEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-6;

            std::swap(bufIn, bufOut);

            // ─────────────────────────────────────────────
            // DISPLAY UPDATE (visualise mode only)
            // Readback is skipped in benchmark mode to
            // keep timing clean.
            // ─────────────────────────────────────────────
            if (visualise && display) {
                const int cellSize = (N <= 100) ? 6 : (N <= 400) ? 2 : 1;
                queue.enqueueReadBuffer(*bufIn, CL_TRUE, 0,
                                        sizeof(int) * gridSize, hostGrid.data());
                CImg<unsigned char> img = gridToImage(hostGrid, N, cellSize);
                display->display(img);
                display->set_title("Forest Fire — Step %d/%d", step + 1, STEPS);
                cimg::wait(200);
            }
        }

            /* ─────────────────────────────────────────────
            // READ BACK & UPDATE DISPLAY EACH STEP
            // Note: readback every step adds overhead —
            // this is display mode only. Benchmarking
            // will skip this (step 7).
            // ─────────────────────────────────────────────
            queue.enqueueReadBuffer(*bufIn, CL_TRUE, 0,
                                    sizeof(int) * gridSize, hostGrid.data());

            img = gridToImage(hostGrid, N, cellSize);
            display.display(img);
            display.set_title("Forest Fire — Step %d/%d", step + 1, STEPS);

            // Delay to make visualisation smoother (recommend 100 for real-time, but 500ms for clearer step-by-step) (remove for benchmarking)
            cimg::wait(100);
        }

        std::cout << "\nFire spread total (" << STEPS << " steps): "
                  << totalSpreadTime << " ms\n";
        std::cout << "Average per step: " << totalSpreadTime / STEPS << " ms\n";

        // Wait for window close before exiting
        while (!display.is_closed()) display.wait();

        // ─────────────────────────────────────────────
        // READ BACK & PRINT FINAL STATE
        // ─────────────────────────────────────────────
        queue.enqueueReadBuffer(*bufIn, CL_TRUE, 0, sizeof(int) * gridSize, hostGrid.data());

        std::cout << "\nFinal grid top-left 10x10:\n";
        for (int y = 0; y < 10; y++) {
            for (int x = 0; x < 10; x++)
                std::cout << hostGrid[y * N + x] << " ";
            std::cout << "\n";
        }
        countCells("Final counts");

        */

            // ─────────────────────────────────────────────
        // TOTAL WALL CLOCK STOP
        // ─────────────────────────────────────────────
        auto wallEnd = std::chrono::steady_clock::now();
        double wallTime = std::chrono::duration<double, std::milli>(wallEnd - wallStart).count();

        // ─────────────────────────────────────────────
        // RESULTS
        // ─────────────────────────────────────────────
        std::cout << "\n--- Timing Results ---\n";
        std::cout << "Init kernel time:        " << initKernelTime << " ms\n";
        std::cout << "Spread total (" << STEPS << " steps): " << totalSpreadTime << " ms\n";
        std::cout << "Spread average per step: " << totalSpreadTime / STEPS << " ms\n";
        std::cout << "Total wall clock time:   " << wallTime << " ms\n";

        // ─────────────────────────────────────────────
        // FINAL CELL COUNTS
        // ─────────────────────────────────────────────
        queue.enqueueReadBuffer(*bufIn, CL_TRUE, 0, sizeof(int) * gridSize, hostGrid.data());
        int empty = 0, trees = 0, burning = 0;
        for (int v : hostGrid) {
            if (v == 0) empty++;
            else if (v == 1) trees++;
            else if (v == 2) burning++;
        }
        std::cout << "\nFinal counts -- Empty: " << empty
                  << "  Trees: " << trees
                  << "  Burning: " << burning << "\n";

        if (visualise && display)
            while (!display->is_closed()) display->wait();

    } catch (const cl::Error& e) {
        std::cerr << "OpenCL error: " << e.what() << " (" << e.err() << ")\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
// main.cpp

#define CL_HPP_TARGET_OPENCL_VERSION 300
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>

std::string loadKernelSource(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open kernel file: " + path);
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
}

int main() {

        // ─────────────────────────────────────────────
        // SIMULATION PARAMETERS
        // ─────────────────────────────────────────────
        const int N               = 100;
        const int STEPS           = 50;
        const float probTree      = 0.8f;
        const float probBurning   = 0.01f;
        const float probImmune    = 0.3f;
        const float probLightning = 0.001f;
        const int gridSize        = N * N;

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

        auto printDevice = [](const std::string& label, const cl::Device& d) {
            std::cout << "\n[" << label << "]\n";
            std::cout << "  Name:          " << d.getInfo<CL_DEVICE_NAME>() << "\n";
            std::cout << "  Compute Units: " << d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << "\n";
            std::cout << "  Global Mem:    " << d.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / (1024*1024) << " MB\n";
            std::cout << "  Max WG Size:   " << d.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << "\n";
        };

        if (foundCPU) printDevice("CPU", cpuDevice);
        if (foundGPU) printDevice("GPU", gpuDevice);

        cl::Device activeDevice = foundGPU ? gpuDevice : cpuDevice;
        std::cout << "\nUsing: " << activeDevice.getInfo<CL_DEVICE_NAME>() << "\n";

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

        double initTime = (initEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                           initEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-6;
        std::cout << "\nGrid init time: " << initTime << " ms\n";

        // ─────────────────────────────────────────────
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

        // ─────────────────────────────────────────────
        // FIRE SPREAD SIMULATION LOOP
        // ─────────────────────────────────────────────
        cl::Kernel spreadKernel(program, "fire_spread");
        cl::Buffer* bufIn  = &bufferA;
        cl::Buffer* bufOut = &bufferB;

        double totalSpreadTime = 0.0;

        for (int step = 0; step < STEPS; step++) {
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
        }

        std::cout << "\nFire spread total (" << STEPS << " steps): " << totalSpreadTime << " ms\n";
        std::cout << "Average per step: " << totalSpreadTime / STEPS << " ms\n";

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

        std::cout << "\nPlaceholder.\n";

}
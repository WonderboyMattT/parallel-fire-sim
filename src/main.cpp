// main.cpp
// Step 3: Grid initialisation kernel

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
    const int N          = 100;   // grid is N×N (try 100, 400, 800 etc later)
    const float probTree     = 0.8f;
    const float probBurning  = 0.01f;
    const float probImmune   = 0.3f;   // not used yet
    const float probLightning = 0.001f; // not used yet
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
    // GRID BUFFER
    // Flat array of N*N ints on the device.
    // CL_MEM_READ_WRITE so kernels can read and write it.
    // ─────────────────────────────────────────────
    int gridSize = N * N;
    cl::Buffer gridBuffer(context, CL_MEM_READ_WRITE, sizeof(int) * gridSize);

    // ─────────────────────────────────────────────
    // INIT GRID KERNEL
    // Seed with current time so each run is different.
    // ─────────────────────────────────────────────
    cl::Kernel initKernel(program, "init_grid");
    cl_ulong seed = (cl_ulong)std::chrono::steady_clock::now().time_since_epoch().count();

    initKernel.setArg(0, gridBuffer);
    initKernel.setArg(1, N);
    initKernel.setArg(2, probTree);
    initKernel.setArg(3, probBurning);
    initKernel.setArg(4, seed);

    // ─────────────────────────────────────────────
    // LAUNCH & PROFILE
    // NDRange is N*N — one work-item per cell.
    // Event lets us measure kernel execution time.
    // ─────────────────────────────────────────────
    cl::Event initEvent;
    queue.enqueueNDRangeKernel(initKernel, cl::NullRange,
                               cl::NDRange(gridSize), cl::NullRange,
                               nullptr, &initEvent);
    queue.finish();

    double initTime = (initEvent.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
                       initEvent.getProfilingInfo<CL_PROFILING_COMMAND_START>()) * 1e-6;
    std::cout << "\nGrid init time: " << initTime << " ms\n";

    // ─────────────────────────────────────────────
    // READ BACK & PRINT A SMALL SAMPLE
    // Read the grid back to host memory to verify
    // the initialisation looks correct.
    // ─────────────────────────────────────────────
    std::vector<int> hostGrid(gridSize);
    queue.enqueueReadBuffer(gridBuffer, CL_TRUE, 0, sizeof(int) * gridSize, hostGrid.data());

    std::cout << "\nTop-left 10x10 of grid (0=empty, 1=tree, 2=burning):\n";
    for (int y = 0; y < 10; y++) {
        for (int x = 0; x < 10; x++) {
            std::cout << hostGrid[y * N + x] << " ";
        }
        std::cout << "\n";
    }

    // Quick sanity check — count cell types
    int empty = 0, trees = 0, burning = 0;
    for (int v : hostGrid) {
        if (v == 0) empty++;
        else if (v == 1) trees++;
        else if (v == 2) burning++;
    }
    std::cout << "\nCell counts — Empty: " << empty
              << "  Trees: " << trees
              << "  Burning: " << burning << "\n";

    std::cout << "\nStep 3 complete. Ready for step 4 (fire spread kernel).\n";
    return 0;
}
#include "src/tensor/tensor.h"
#include <iostream>
#include <chrono>
#include <thread>

int main() {
    std::cout << "[INFO] Creating a 4x4 tensor on device...\n";
    CudaTensor tensor({4, 4});

    std::cout << "[INFO] Tensor allocated!\n";
    // tensor.debug();  // Optional: print internal data or info

    std::cout << "[INFO] Running a fake compute task...\n";

    // Simulate CPU work for 5 seconds so it shows in top/htop
    volatile long dummy = 0;
    auto start = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count() < 5.0) {
        for (int i = 0; i < 1e6; ++i) dummy += i;
    }

    std::cout << "[INFO] Done.\n";
    return 0;
}

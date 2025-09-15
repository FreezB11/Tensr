#pragma once
#include <chrono>
#include <iostream>
#include <string>

class Timer {
public:
    Timer(const std::string& name = "")
        : name_(name), start_(std::chrono::high_resolution_clock::now()) {}

    ~Timer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end - start_).count();
        if (!name_.empty()) {
            std::cout << name_ << " ";
        }
        std::cout << "[" << duration << " us]" << std::endl;
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};
/// @file: tensr.h
#pragma once
#include "device.h"
#include <vector>
#include <memory>

#include <random>

/// @class tensr
template <typename T>
class tensr{
private:
    // we will change this in the future
    std::shared_ptr<T> data;
    std::vector<int> shape;
    // to be made
    Device device;
    size_t noe; // no of elements
public:
    // here to we shall decide
    bool toGPU();
    bool toCPU();
    tensr(std::vector<int>& shape, std::vector<T>& data);
    explicit tensr(std::vector<int>& shape);
    explicit tensr(std::vector<int>& shape, bool rand);
    // ~tensr();
    void print();
};
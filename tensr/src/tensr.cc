/// @file: tensr.cc
#include "tensr.h"
#include "device.h"
#include <iostream>
/// @def tensr class
template <typename T>
tensr<T>::tensr(std::vector<int>& shape, std::vector<T>& data){
    noe = 1;
    for(int dim : shape){
        noe *= dim;
    }

    if(noe != data.size()){
        std::__throw_invalid_argument("buddy look at your data size it is either less or more fix that");
    }

    T* raw_ptr = new T[noe];

    for(size_t i = 0; i < noe; i++){
        raw_ptr[i] = data[i];
    }

    this->data = std::shared_ptr<T>(raw_ptr, std::default_delete<T[]>());
};

template <typename T>
tensr<T>::tensr(std::vector<int>& shape){
    noe = 1;
    for(int dim : shape){
        noe *= dim;
    }
    T* raw_ptr = new T[noe];

    this->data = std::shared_ptr<T>(raw_ptr, std::default_delete<T[]>());
}

template <typename T>
tensr<T>::tensr(std::vector<int>& shape, bool rand){
    noe = 1;
    for(int dim : shape){
        noe *= dim;
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<T> dist(0.0f, 1.0f);

    T* raw_ptr = new T[noe];

    for(int i = 0; i < noe; i++){
        raw_ptr[i] = dist(gen);
    }

    this->data = std::shared_ptr<T>(raw_ptr, std::default_delete<T[]>());
}

template <typename T>
void tensr<T>::print(){
    std::cout << "Tensor (Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i == shape.size() - 1 ? "" : ", ");
    }
    std::cout << "])" << std::endl;
    // Use .get() to access the raw pointer for reading
    float* raw_data = data.get();
    std::cout << "Data: [ ";
    for (size_t i = 0; i < noe; ++i) {
        std::cout << raw_data[i] << " ";
    }
    std::cout << "]" << std::endl;
}

// At the end of tensr.cc
template class tensr<float>;

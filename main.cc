#include <iostream>
#include "tensr.h"
#include <vector>

int main(){
    std::vector<float> data = {1,1,1,
                                2,2,2,
                                3,3,3,
                               1,0,0,
                                0,1,0,
                                0,0,1};
    std::vector<int> shape = {2,3,3};
    tensr<float> test(shape, data);
    test.print();
    return 0;
}
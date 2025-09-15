#include <immintrin.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include "../utils/timer.h"

typedef __m256 m32;

// 8x8 matrix using AVX (row-major, 1 __m256 per row)
struct M88 {
    m32 rows[8];

    M88() {
        for (int i = 0; i < 8; i++)
            rows[i] = _mm256_setzero_ps();
    }

    void load(const float* data, int stride) {
        for (int i = 0; i < 8; ++i)
            rows[i] = _mm256_load_ps(&data[i * stride]);
    }

    void store(float* out, int stride) const {
        for (int i = 0; i < 8; ++i)
            _mm256_store_ps(&out[i * stride], rows[i]);
    }

    // Matrix multiplication: C = A * B
    M88 operator*(const M88* B) const {
        M88 C;
        for (int i = 0; i < 8; i++) {
            m32 sum = _mm256_setzero_ps();
            for (int k = 0; k < 8; k++) {
                m32 a = _mm256_set1_ps(((float*)&rows[i])[k]);
                m32 b = B->rows[k];
                sum = _mm256_fmadd_ps(a, b, sum);
            }
            C.rows[i] = sum;
        }
        return C;
    }

    M88 operator+(const M88& other) const {
        M88 out;
        for (int i = 0; i < 8; ++i)
            out.rows[i] = _mm256_add_ps(rows[i], other.rows[i]);
        return out;
    }
};

// Load 8x8 block from large matrix
M88 load_block(const float* matrix, int row, int col, int N) {
    M88 block;
    for (int i = 0; i < 8; ++i)
        block.rows[i] = _mm256_load_ps(&matrix[(row + i) * N + col]);
    return block;
}

// Store 8x8 block into large matrix
void store_block(float* matrix, const M88& block, int row, int col, int N) {
    for (int i = 0; i < 8; ++i)
        _mm256_store_ps(&matrix[(row + i) * N + col], block.rows[i]);
}

// Multiply N x N matrices using 8x8 SIMD blocks (parallelized)
void matrix_mul_simd(const float* A, const float* B, float* C, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += 8) {
        for (int j = 0; j < N; j += 8) {
            M88 result;
            for (int k = 0; k < N; k += 8) {
                M88 A_block = load_block(A, i, k, N);
                M88 B_block = load_block(B, k, j, N);
                result = result + (A_block * &B_block);
            }
            store_block(C, result, i, j, N);
        }
    }
}

// Initialize matrix with random floats
void fill_random(float* mat, int N) {
    for (int i = 0; i < N * N; ++i)
        mat[i] = static_cast<float>(rand()) / RAND_MAX;
}

// Print small matrix (for debug)
void print_matrix(const float* mat, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j)
            std::cout << mat[i * N + j] << " ";
        std::cout << "\n";
    }
}

int main() {
    constexpr int N = 128; // must be divisible by 8
    float* A = (float*)aligned_alloc(32, sizeof(float) * N * N);
    float* B = (float*)aligned_alloc(32, sizeof(float) * N * N);
    float* C = (float*)aligned_alloc(32, sizeof(float) * N * N);
    std::memset(C, 0, sizeof(float) * N * N);

    srand((unsigned)time(0));
    fill_random(A, N);
    fill_random(B, N);

    {
        Timer T;
        matrix_mul_simd(A, B, C, N);
    }

    // Debug output if needed:
    // std::cout << "Matrix A:\n"; print_matrix(A, N);
    // std::cout << "Matrix B:\n"; print_matrix(B, N);
    // std::cout << "Matrix C = A * B:\n"; print_matrix(C, N);

    free(A); free(B); free(C);
    return 0;
}
// yashr@HSAY:~/doc/dump/simd/threads$ g++ -O3 -mavx2 -mfma -march=native -fopenmp -std=c++17 gpt2.cc -o simd_matmul && ./simd_matmul 

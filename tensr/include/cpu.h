#include <immintrin.h>
#include <omp.h>
#include <iostream>

typedef __m256 m32;

struct m88{
    m32 rows[8];
    m88(){
        for(int i = 0; i < 8; i++){
            rows[i] = _mm256_setzero_ps();
        }
    }
    void load(const float* data, int stride){
        for(int i= 0; i < 8; i++){
            rows[i] = _mm256_load_ps(&data[i*stride]);
        }
    }
    void store(float* out, int stride) const{
        for(int i = 0; i < 8; i++)
            _mm256_store_ps(&out[i*stride], rows[i]);
    }

    m88 operator*(const m88* B) const{
        m88 C;
        for(int i = 0; i < 8; i++){
            m32 sum = _mm256_setzero_ps();
            for(int k = 0; k < 8; k++){
                m32 a = _mm256_set1_ps(((float*)&rows[i])[k]);
                m32 b = B->rows[k];
                sum = _mm256_fmadd_ps(a,b, sum);
            }
            C.rows[i] = sum;
        }
        return C;
    }

    m88 operator+(const m88& other) const {
        m88 out;
        for (int i = 0; i < 8; ++i)
            out.rows[i] = _mm256_add_ps(rows[i], other.rows[i]);
        return out;
    }
};


m88 load_block(const float* matrix, int row, int col, int N) {
    m88 block;
    for (int i = 0; i < 8; ++i)
        block.rows[i] = _mm256_load_ps(&matrix[(row + i) * N + col]);
    return block;
}

// Store 8x8 block into large matrix
void store_block(float* matrix, const m88& block, int row, int col, int N) {
    for (int i = 0; i < 8; ++i)
        _mm256_store_ps(&matrix[(row + i) * N + col], block.rows[i]);
}

// Multiply N x N matrices using 8x8 SIMD blocks (parallelized)
void matrix_mul_simd(const float* A, const float* B, float* C, int N) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i += 8) {
        for (int j = 0; j < N; j += 8) {
            m88 result;
            for (int k = 0; k < N; k += 8) {
                m88 A_block = load_block(A, i, k, N);
                m88 B_block = load_block(B, k, j, N);
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
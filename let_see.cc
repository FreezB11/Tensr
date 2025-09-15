#include <immintrin.h>
#include "../utils/timer.h"
#include <iostream>
#include <random>
/*
    LOAD && STORE
    | Intrinsic                                          | Description                         |
    | -------------------------------------------------- | ----------------------------------- |
    | `__m256 _mm256_setzero_ps(void)`                   | Set all elements to `0.0f`          |
    | `__m256 _mm256_set1_ps(float a)`                   | Broadcast single float to all 8     |
    | `__m256 _mm256_set_ps(f7, ..., f0)`                | Set all 8 floats individually       |
    | `__m256 _mm256_load_ps(float const* mem_addr)`     | Load 8 floats from aligned memory   |
    | `__m256 _mm256_loadu_ps(float const* mem_addr)`    | Load 8 floats from unaligned memory |
    | `void _mm256_store_ps(float* mem_addr, __m256 a)`  | Store 8 floats to aligned memory    |
    | `void _mm256_storeu_ps(float* mem_addr, __m256 a)` | Store 8 floats to unaligned memory  |

    ARITHMETIC OPS
    | Intrinsic                                  | Description           |
    | ------------------------------------------ | --------------------- |
    | `__m256 _mm256_add_ps(__m256 a, __m256 b)` | Element-wise addition |
    | `__m256 _mm256_sub_ps(__m256 a, __m256 b)` | Subtraction           |
    | `__m256 _mm256_mul_ps(__m256 a, __m256 b)` | Multiplication        |
    | `__m256 _mm256_div_ps(__m256 a, __m256 b)` | Division              |
    | `__m256 _mm256_sqrt_ps(__m256 a)`          | Square root           |

    COMPARISON OPS
    | Intrinsic                                              | Description           |
    | ------------------------------------------------------ | --------------------- |
    | `__m256 _mm256_cmp_ps(__m256 a, __m256 b, int cmp_op)` | General comparison    |
    | `__m256 _mm256_cmp_eq_ps(a, b)`                        | Equal (`a == b`)      |
    | `__m256 _mm256_cmp_lt_ps(a, b)`                        | Less than (`a < b`)   |
    | `__m256 _mm256_cmp_le_ps(a, b)`                        | Less than or equal    |
    | `__m256 _mm256_cmp_gt_ps(a, b)`                        | Greater than          |
    | `__m256 _mm256_cmp_ge_ps(a, b)`                        | Greater than or equal |
    | `__m256 _mm256_cmp_neq_ps(a, b)`                       | Not equal             |

    LOGICAL / BITWISE
    | Intrinsic                                     | Description                      |
    | --------------------------------------------- | -------------------------------- |
    | `__m256 _mm256_and_ps(__m256 a, __m256 b)`    | Bitwise AND                      |
    | `__m256 _mm256_or_ps(__m256 a, __m256 b)`     | Bitwise OR                       |
    | `__m256 _mm256_xor_ps(__m256 a, __m256 b)`    | Bitwise XOR                      |
    | `__m256 _mm256_andnot_ps(__m256 a, __m256 b)` | Bitwise AND NOT (i.e., `~a & b`) |

    MATH APPROX
    | Intrinsic                          | Description                        |
    | ---------------------------------- | ---------------------------------- |
    | `__m256 _mm256_rcp_ps(__m256 a)`   | Approximate reciprocal             |
    | `__m256 _mm256_rsqrt_ps(__m256 a)` | Approximate reciprocal square root |

    BLENDING & MASKING
    | Intrinsic                                                           | Description                          |
    | ------------------------------------------------------------------- | ------------------------------------ |
    | `__m256 _mm256_blend_ps(__m256 a, __m256 b, const int imm8)`        | Blend based on mask bits             |
    | `__m256 _mm256_blendv_ps(__m256 a, __m256 b, __m256 mask)`          | Blend with vector mask               |
    | `__m256 _mm256_permute_ps(__m256 a, const int imm8)`                | Shuffle float elements within lanes  |
    | `__m256 _mm256_permute2f128_ps(__m256 a, __m256 b, const int imm8)` | Shuffle 128-bit lanes across vectors |

    HORIZONTAL OPS
    | Intrinsic                                   | Description                                              |
    | ------------------------------------------- | -------------------------------------------------------- |
    | `float _mm256_reduce_add_ps(__m256 a)`      | No built-in version before AVX-512, must reduce manually |
    | `__m256 _mm256_hadd_ps(__m256 a, __m256 b)` | Horizontal add pairs                                     |
    | `__m256 _mm256_hsub_ps(__m256 a, __m256 b)` | Horizontal subtract pairs                                |

    CONVERSION
    | Intrinsic                              | Description                   |
    | -------------------------------------- | ----------------------------- |
    | `__m256i _mm256_cvtps_epi32(__m256 a)` | Convert floats to 32-bit ints |
    | `__m256 _mm256_cvtepi32_ps(__m256i a)` | Convert 32-bit ints to floats |

*/
class M88 {
public:
    __m256 rows[8];  // Each row is 8 floats packed into __m256
    M88() = default;
    M88(float value) {
        __m256 val = _mm256_set1_ps(value);
        for (int i = 0; i < 8; ++i) {
            rows[i] = val;
        }
    }
    M88 operator+(const M88& other) const {
        M88 result;
        for (int i = 0; i < 8; ++i) {
            result.rows[i] = _mm256_add_ps(rows[i], other.rows[i]);
        }
        return result;
    }
    /*
        this works better because of the cache locality
        so the read is faster than the other where the 
        data is stored in the memory line and the access
         is slow
    */
    M88 operator*(const M88* other) const {
        M88 result;

        // First, transpose the right-hand-side matrix to access columns easily
        alignas(32) float other_data[8][8];
        for (int i = 0; i < 8; ++i) {
            _mm256_store_ps(other_data[i], other->rows[i]);
        }

        for (int i = 0; i < 8; ++i) {
            alignas(32) float row_data[8];
            _mm256_store_ps(row_data, this->rows[i]);

            alignas(32) float res_row[8];

            for (int j = 0; j < 8; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < 8; ++k) {
                    sum += row_data[k] * other_data[k][j];
                }
                res_row[j] = sum;
            }

            result.rows[i] = _mm256_load_ps(res_row);
        }

        return result;
    }
    static float horizontal_sum(__m256 v) {
        __m128 low = _mm256_castps256_ps128(v);
        __m128 high = _mm256_extractf128_ps(v, 1);
        __m128 sum = _mm_add_ps(low, high);
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    }
    // SIMD matrix multiply
    // M88 operator*(const M88* other) const {
    //     M88 result;

    //     // Need to access columns from 'other'
    //     alignas(32) float other_matrix[8][8];
    //     for (int i = 0; i < 8; ++i) {
    //         _mm256_store_ps(other_matrix[i], other->rows[i]);
    //     }

    //     for (int i = 0; i < 8; ++i) {
    //         float row_result[8];
    //         for (int j = 0; j < 8; ++j) {
    //             // Build column j of 'other' as __m256
    //             alignas(32) float col_vals[8];
    //             for (int k = 0; k < 8; ++k) {
    //                 col_vals[k] = other_matrix[k][j];
    //             }
    //             __m256 col = _mm256_load_ps(col_vals);

    //             // Dot product of this->rows[i] and col
    //             __m256 mul = _mm256_mul_ps(rows[i], col);
    //             row_result[j] = horizontal_sum(mul); // SIMD sum
    //         }
    //         result.rows[i] = _mm256_load_ps(row_result);
    //     }

    //     return result;
    // }
    // Print matrix (for debugging)
    void print() const {
        for (int i = 0; i < 8; ++i) {
            alignas(32) float vals[8];
            _mm256_store_ps(vals, rows[i]);
            for (int j = 0; j < 8; ++j) {
                std::cout << vals[j] << " ";
            }
            std::cout << "\n";
        }
    }
    // Constructor with random values
    static M88 Random() {
        M88 mat;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int i = 0; i < 8; ++i) {
            alignas(32) float values[8];
            for (int j = 0; j < 8; ++j) {
                values[j] = dist(gen);
            }
            mat.rows[i] = _mm256_load_ps(values);
        }

        return mat;
    }
};


// int main() {
//     M88 A(1.0f);  // Fill with 1s
//     M88 B(2.0f);  // Fill with 2s

//     M88 C = A + B;

//     std::cout << "Matrix C:\n";
//     C.print();
// }
int main() {
    // M88 A(1.0f);  // all ones
    // M88 B(2.0f);  // all twos
    M88 A = M88::Random();
    M88 B = M88::Random();
    M88 C;
    {  
        Timer T;
        C = A * &B;
    }
    std::cout << "Result:\n";
    C.print();  // Should print 16 in all cells (8 × 2)
}


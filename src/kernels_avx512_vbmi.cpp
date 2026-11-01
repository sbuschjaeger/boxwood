// The avx512_vbmi backend: the AVX-512 kernels with VBMI byte permutes and
// VNNI dot products in the int8 path. CMake compiles this file with the
// avx512 flags plus -mavx512vbmi -mavx512vnni.

#define BOXWOOD_AVX512_VBMI 1
#include "kernels_avx512.cpp"

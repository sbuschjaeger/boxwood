// The backend table and the run-time CPU check. The CPU check is the only
// compiler-specific code outside the SIMD backend files.

#include "backend.hpp"

#include <string>
#include <vector>

#if defined(BOXWOOD_SIMD_x86) && defined(_MSC_VER)
#include <intrin.h>
#endif

namespace boxwood {
namespace {

bool scalar_available() { return true; }

// The build defines BOXWOOD_SIMD_x86, BOXWOOD_SIMD_neon or BOXWOOD_SIMD_none
// for the target processor and compiles only that target's kernel files.
#if defined(BOXWOOD_SIMD_x86) && defined(_MSC_VER)
/// Whether the CPU has all `ebx` and `ecx` bits of cpuid leaf 7 (subleaf 0)
/// and the OS saves the register state bits `xcr0` (so it keeps the vector
/// registers across context switches).
bool cpu_has(unsigned ebx, unsigned ecx, unsigned long long xcr0) {
  int info[4];
  __cpuid(info, 1);
  if ((info[2] & (1 << 27)) == 0 || (_xgetbv(0) & xcr0) != xcr0) return false;  // 27 = OSXSAVE
  __cpuidex(info, 7, 0);
  return (static_cast<unsigned>(info[1]) & ebx) == ebx && (static_cast<unsigned>(info[2]) & ecx) == ecx;
}

// XCR0: 0x6 = the SSE and AVX state, 0xE6 = also the AVX-512 state.
bool avx2_available() { return cpu_has(1u << 5, 0, 0x6); }
// EBX: 16 = AVX512F, 30 = AVX512BW, 31 = AVX512VL.
bool avx512_available() { return cpu_has((1u << 16) | (1u << 30) | (1u << 31), 0, 0xE6); }
// ECX: 1 = AVX512_VBMI, 11 = AVX512_VNNI.
bool avx512_vbmi_available() {
  return cpu_has((1u << 16) | (1u << 30) | (1u << 31), (1u << 1) | (1u << 11), 0xE6);
}
#elif defined(BOXWOOD_SIMD_x86)
// __builtin_cpu_supports also checks that the OS saves the register state.
bool avx2_available() { return __builtin_cpu_supports("avx2") != 0; }
bool avx512_available() {
  return __builtin_cpu_supports("avx512f") != 0 && __builtin_cpu_supports("avx512bw") != 0 &&
         __builtin_cpu_supports("avx512vl") != 0;
}
bool avx512_vbmi_available() {
  return avx512_available() && __builtin_cpu_supports("avx512vbmi") != 0 &&
         __builtin_cpu_supports("avx512vnni") != 0;
}
#elif defined(BOXWOOD_SIMD_neon)
bool neon_available() { return true; }  // part of every AArch64 CPU
#endif

// Fastest first; the last entry is always available.
const Backend kBackends[] = {
#if defined(BOXWOOD_SIMD_x86)
    {"avx512_vbmi", avx512_vbmi_available, 64, avx512_vbmi::encode_numeric, avx512_vbmi::accumulate,
     avx512_vbmi::leaf_index, avx512_vbmi::ctr_lookup},
    {"avx512", avx512_available, 64, avx512::encode_numeric, avx512::accumulate, avx512::leaf_index,
     scalar::ctr_lookup},
    {"avx2", avx2_available, 32, avx2::encode_numeric, avx2::accumulate, avx2::leaf_index, scalar::ctr_lookup},
#elif defined(BOXWOOD_SIMD_neon)
    {"neon", neon_available, 16, scalar::encode_numeric, neon::accumulate, neon::leaf_index, scalar::ctr_lookup},
#endif
    {"scalar", scalar_available, 1, scalar::encode_numeric, scalar::accumulate, scalar::leaf_index,
     scalar::ctr_lookup},
};

}  // namespace

const Backend& best_backend() {
  static const Backend& best = [] {
    for (const Backend& b : kBackends) {
      if (b.available()) return b;
    }
    return kBackends[0];  // unreachable: the last backend is always available
  }();
  return best;
}

const Backend* find_backend(std::string_view name) {
  for (const Backend& b : kBackends) {
    if (name == b.name) return b.available() ? &b : nullptr;
  }
  return nullptr;
}

std::vector<std::string> available_backends() {
  std::vector<std::string> out;
  for (const Backend& b : kBackends) {
    if (b.available()) out.push_back(b.name);
  }
  return out;
}

}  // namespace boxwood

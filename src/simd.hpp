#pragma once

// Helpers for the SIMD backend files. Those files must not call inline or
// template functions with external linkage (std::vector, std::min, ...): the
// compiler may emit such a function compiled with the file's SIMD flags, the
// linker keeps one copy of it for the whole program, and that copy can then
// run on a CPU without those instructions. Everything here has internal
// linkage, so each file has its own copy. A ctest checks the object files.

#include <cstddef>
#include <cstdint>

namespace boxwood {
namespace {

/// A zero-initialized heap array of n values.
template <class T>
class Scratch {
 public:
  explicit Scratch(size_t n) : p_(new T[n]()) {}
  ~Scratch() { delete[] p_; }
  Scratch(const Scratch&) = delete;
  Scratch& operator=(const Scratch&) = delete;
  T* data() { return p_; }
  T& operator[](size_t i) { return p_[i]; }

 private:
  T* p_;
};

constexpr uint32_t min_u32(uint32_t a, uint32_t b) { return a < b ? a : b; }

}  // namespace
}  // namespace boxwood

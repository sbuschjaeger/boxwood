// A port of city_hash64 from tests/catboost_ref.py (Yandex's CityHash variant).
// All arithmetic is unsigned 64-bit and wraps mod 2^64, as in the reference.

#include "cityhash.hpp"

#include <cstring>

static_assert(std::endian::native == std::endian::little, "boxwood loads little-endian bytes");

namespace boxwood {
namespace {

constexpr uint64_t kK0 = 0xC3A5C85C97CB3127ULL;
constexpr uint64_t kK1 = 0xB492B66FBE98F273ULL;
constexpr uint64_t kK2 = 0x9AE16A3B2F90404FULL;
constexpr uint64_t kK3 = 0xC949D7C7509E6557ULL;
constexpr uint64_t kMul = 0x9DDFEA08EB382D69ULL;

inline uint64_t rot(uint64_t v, uint32_t s) { return (v >> s) | (v << (64 - s)); }

inline uint64_t shift_mix(uint64_t v) { return v ^ (v >> 47); }

inline uint64_t load64(const char* s, size_t i) {
  uint64_t v;
  std::memcpy(&v, s + i, 8);
  return v;
}

inline uint64_t load32(const char* s, size_t i) {
  uint32_t v;
  std::memcpy(&v, s + i, 4);
  return v;
}

inline uint8_t byte_at(const char* s, size_t i) { return static_cast<uint8_t>(s[i]); }

uint64_t hash_len16(uint64_t u, uint64_t v) {
  const uint64_t a = shift_mix((u ^ v) * kMul);
  const uint64_t b = shift_mix((v ^ a) * kMul);
  return b * kMul;
}

uint64_t hash_len0to16(const char* s, uint64_t n) {
  if (n > 8) {
    const uint64_t a = load64(s, 0);
    const uint64_t tail = load64(s, n - 8);
    return hash_len16(a, rot(tail + n, static_cast<uint32_t>(n))) ^ tail;
  }
  if (n >= 4) {
    const uint64_t a = load32(s, 0);
    return hash_len16(n + (a << 3), load32(s, n - 4));
  }
  if (n > 0) {
    const uint64_t y = byte_at(s, 0) + (byte_at(s, n >> 1) << 8);
    const uint64_t z = n + (byte_at(s, n - 1) << 2);
    return shift_mix((y * kK2) ^ (z * kK3)) * kK2;
  }
  return kK2;
}

uint64_t hash_len17to32(const char* s, uint64_t n) {
  const uint64_t a = load64(s, 0) * kK1;
  const uint64_t x = load64(s, 8);
  const uint64_t c = load64(s, n - 8) * kK2;
  const uint64_t d = load64(s, n - 16) * kK0;
  return hash_len16(rot(a - x, 43) + rot(c, 30) + d, a + rot(x ^ kK3, 20) - c + n);
}

struct Pair {
  uint64_t first;
  uint64_t second;
};

/// CityHash's WeakHashLen32WithSeeds of the 32 bytes at `s[i]`.
Pair weak_hash_at(const char* s, size_t i, uint64_t a, uint64_t b) {
  const uint64_t w = load64(s, i);
  const uint64_t x = load64(s, i + 8);
  const uint64_t y = load64(s, i + 16);
  const uint64_t z = load64(s, i + 24);
  a += w;
  b = rot(b + a + z, 21);
  const uint64_t c = a;
  a += x + y;
  b += rot(a, 44);
  return {a + z, b + c};
}

uint64_t hash_len33to64(const char* s, uint64_t n) {
  const uint64_t z0 = load64(s, 24);
  uint64_t a = load64(s, 0) + (n + load64(s, n - 16)) * kK0;
  uint64_t x = rot(a + z0, 52);
  uint64_t c = rot(a, 37);
  a += load64(s, 8);
  c += rot(a, 7);
  a += load64(s, 16);
  const uint64_t vf = a + z0;
  const uint64_t vs = x + rot(a, 31) + c;

  a = load64(s, 16) + load64(s, n - 32);
  const uint64_t z = load64(s, n - 8);
  x = rot(a + z, 52);
  c = rot(a, 37);
  a += load64(s, n - 24);
  c += rot(a, 7);
  a += load64(s, n - 16);
  const uint64_t wf = a + z;
  const uint64_t ws = x + rot(a, 31) + c;

  const uint64_t r = shift_mix((vf + ws) * kK2 + (wf + vs) * kK0);
  return shift_mix(r * kK0 + vs) * kK2;
}

}  // namespace

uint64_t city_hash64(std::string_view s) {
  const char* d = s.data();
  const uint64_t n = s.size();
  if (n <= 32) return n <= 16 ? hash_len0to16(d, n) : hash_len17to32(d, n);
  if (n <= 64) return hash_len33to64(d, n);

  uint64_t x = load64(d, 0);
  uint64_t y = load64(d, n - 16) ^ kK1;
  uint64_t z = load64(d, n - 56) ^ kK0;
  Pair v = weak_hash_at(d, n - 64, n, y);
  Pair w = weak_hash_at(d, n - 32, n * kK1, kK0);
  z += shift_mix(v.second) * kK1;
  x = rot(z + x, 39) * kK1;
  y = rot(y, 33) * kK1;
  // The loop runs over the first (n - 1) rounded down to a multiple of 64 bytes;
  // the last 64 bytes are already folded into v, w, x, y, z above.
  for (size_t i = 0, end = static_cast<size_t>((n - 1) / 64) * 64; i < end; i += 64) {
    x = rot(x + y + v.first + load64(d, i + 16), 37) * kK1;
    y = rot(y + v.second + load64(d, i + 48), 42) * kK1;
    x ^= w.second;
    y ^= v.first;
    z = rot(z ^ w.first, 33);
    v = weak_hash_at(d, i, v.second * kK1, x + w.first);
    w = weak_hash_at(d, i + 32, z + w.second, y);
    std::swap(z, x);
  }
  return hash_len16(hash_len16(v.first, w.first) + shift_mix(y) * kK1 + z,
                    hash_len16(v.second, w.second) + x);
}

uint32_t cat_hash(std::string_view s) { return static_cast<uint32_t>(city_hash64(s)); }

}  // namespace boxwood

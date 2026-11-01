#pragma once

#include <stdexcept>
#include <string>

namespace boxwood {

/// The only error type the library throws.
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace boxwood

/// The only way to report an error. Use at the API boundary (import, from_bytes,
/// input shapes); kernels and hot loops never check and never throw.
/// `msg` must convert to std::string.
#define BW_CHECK(cond, msg) \
  do {                     \
    if (!(cond)) throw ::boxwood::Error(msg); \
  } while (0)

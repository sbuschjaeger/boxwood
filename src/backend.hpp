#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "kernels.hpp"

namespace boxwood {

/// One prediction backend: the block kernels plus the run-time CPU check.
/// Each backend is one .cpp file compiled with its own flags; the codes of a
/// block are feature-major (see OTEView).
struct Backend {
  const char* name;
  bool (*available)();  // CPU check at run time
  size_t group_size;    // observations per vector group of the kernels (1 = scalar)
  void (*encode_numeric)(const OTEView&, const float* X, size_t n_obs, uint8_t* codes);
  void (*accumulate)(const OTEView&, const uint8_t* codes, size_t n_obs, double* margins);
  void (*leaf_index)(const OTEView&, const uint8_t* codes, size_t n_obs, uint8_t* out);
  scalar::CtrLookup ctr_lookup;
};

/// The first available backend of the table (fastest first), cached after the
/// first call. The table ends with `scalar`, which is always available.
const Backend& best_backend();

/// The backend named `name`, or nullptr if the name is unknown or the backend
/// is not available on this CPU.
const Backend* find_backend(std::string_view name);

}  // namespace boxwood

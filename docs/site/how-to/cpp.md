# Use boxwood from C++

The library is C++20 with no dependencies beyond the standard library. Everything public is in
`include/boxwood/ensemble.hpp` (the `ObliviousTreeEnsemble` class and the `Batch` of observations) and
`include/boxwood/error.hpp` (`boxwood::Error`).

## Add it to a CMake project

```cmake
set(BOXWOOD_PYTHON OFF CACHE BOOL "" FORCE)   # the library only, no Python module
add_subdirectory(boxwood EXCLUDE_FROM_ALL)
target_link_libraries(my_app PRIVATE boxwood)
```

`boxwood` is a static library target with its include directory attached. The SIMD backends are compiled with
their own flags and chosen at run time, so the rest of your project needs no special flags.

## Load a model and predict

Export the CatBoost model as JSON in Python (`model.save_model("model.json", format="json")`), or load a model
saved by boxwood's `to_bytes` with `ObliviousTreeEnsemble::from_bytes`.

```cpp
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string_view>
#include <vector>

#include "boxwood/ensemble.hpp"

int main(int argc, char** argv) {
  // A model saved by CatBoost: model.save_model("model.json", format="json").
  std::ifstream file(argv[1]);
  std::stringstream text;
  text << file.rdbuf();
  auto ote = boxwood::ObliviousTreeEnsemble::from_catboost_json(text.str());
  const boxwood::Info info = ote.info();
  std::printf("%u trees, %u numeric and %u categorical features\n", info.n_trees, info.n_numeric,
              info.n_categorical);

  // Two observations: the numeric values observation-major (NaN = missing), then one column of
  // values per categorical feature.
  const size_t n = 2;
  std::vector<float> numeric(n * info.n_numeric, 0.5f);
  std::vector<std::vector<std::string_view>> columns(info.n_categorical, {"v1", "v7"});
  boxwood::Batch batch{numeric, n};
  for (const auto& col : columns) batch.categorical.emplace_back(col);

  std::vector<double> margin(n * info.n_outputs);
  ote.predict_margin(batch, margin);
  std::printf("margins %.6f %.6f\n", margin[0], margin[1]);

  // The same observations dictionary-encoded: each distinct value is hashed once.
  const std::vector<std::string_view> dictionary = {"v1", "v7"};
  const std::vector<uint32_t> codes = {0, 1};
  boxwood::Batch encoded{numeric, n};
  for (uint32_t c = 0; c < info.n_categorical; ++c) {
    encoded.categorical.emplace_back(dictionary);
    encoded.categorical_codes.emplace_back(codes);
  }
  std::vector<double> again(n * info.n_outputs);
  ote.set_threads(4);
  ote.predict_margin(encoded, again);
  std::printf("same: %s\n", again == margin ? "yes" : "no");

  // Errors are boxwood::Error (a std::runtime_error).
  try {
    ote.predict_margin(boxwood::Batch{numeric, n + 1}, margin);
  } catch (const boxwood::Error& e) {
    std::printf("error: %s\n", e.what());
  }
}
```

With the categorical benchmark model this prints:

```text
1000 trees, 20 numeric and 4 categorical features
margins -0.790378 1.213910
same: yes
error: X.numeric size does not match n * n_numeric
```

## The Batch

| field | content |
|---|---|
| `numeric` | n × n_numeric float32 values, observation-major; NaN = missing |
| `n` | the number of observations |
| `categorical` | one span of n values per categorical feature (CatBoost's order), or one dictionary per feature |
| `y`, `weight` | the target and the observation weights, for `fit` and `update` only |
| `categorical_codes` | optional: one span of n codes per categorical feature into its dictionary |

The spans point at your memory; the `Batch` copies nothing. `info().numeric_flat_index` and
`categorical_flat_index` give the CatBoost column of every numeric and categorical feature, so you can pick the
columns out of a wider table.

## Errors

Every check happens at the API boundary (import, `from_bytes`, the shapes of the inputs) and throws
`boxwood::Error`, a `std::runtime_error`. The kernels do not check and do not throw.

## Threads and backends

`set_threads(n)` and `set_backend("avx2")` work as in Python (see [Fast inference](../tutorials/fast-inference.md)).
A const `ObliviousTreeEnsemble` can be shared between threads for prediction.

The full C++ API is in the [C++ reference](../reference/cpp.md).

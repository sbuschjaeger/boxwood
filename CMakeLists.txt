cmake_minimum_required(VERSION 3.21)
project(boxwood CXX)

option(BOXWOOD_SANITIZE "Add -fsanitize=address,undefined (GCC/Clang only)" OFF)
option(BOXWOOD_PYTHON "Build the Python extension module boxwood._boxwood" ON)
option(BOXWOOD_TESTS "Build the C++ tests" ON)

# The SIMD backends of the target: x86-64 has AVX2 and AVX-512 (checked at run
# time), AArch64 always has NEON. backend.cpp lists only the compiled ones.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
  set(BOXWOOD_SIMD x86)
  set(BOXWOOD_SIMD_SOURCES src/kernels_avx2.cpp src/kernels_avx512.cpp src/kernels_avx512_vbmi.cpp)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64|ARM64)$")
  set(BOXWOOD_SIMD neon)
  set(BOXWOOD_SIMD_SOURCES src/kernels_neon.cpp)
else()
  set(BOXWOOD_SIMD none)
  set(BOXWOOD_SIMD_SOURCES)
endif()

add_library(boxwood STATIC
  src/cityhash.cpp
  src/json.cpp
  src/backend.cpp
  src/kernels_scalar.cpp
  ${BOXWOOD_SIMD_SOURCES}
  src/categorical.cpp
  src/fit.cpp
  src/prune.cpp
  src/ensemble.cpp
  src/ensemble_fit.cpp
  src/import_catboost.cpp
  src/serialize.cpp)
target_compile_features(boxwood PUBLIC cxx_std_20)
find_package(Threads REQUIRED)
target_link_libraries(boxwood PUBLIC Threads::Threads)
set_target_properties(boxwood PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(boxwood PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_source_files_properties(src/backend.cpp PROPERTIES COMPILE_DEFINITIONS "BOXWOOD_SIMD_${BOXWOOD_SIMD}")
# The SIMD backends are the only files compiled with SIMD flags; they are
# selected at run time by the CPU check in backend.cpp. -ffp-contract=off
# keeps the int8 leaf multiply and the block add two separate operations, so
# every backend gives the same bits (MSVC's default /fp:precise does not
# contract).
if(MSVC)
  set_source_files_properties(src/kernels_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  set_source_files_properties(src/kernels_avx512.cpp src/kernels_avx512_vbmi.cpp
    PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
  set_source_files_properties(src/kernels_scalar.cpp src/kernels_neon.cpp PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
  set_source_files_properties(src/kernels_avx2.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx2;-mfma;-ffp-contract=off")
  set_source_files_properties(src/kernels_avx512.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx2;-mfma;-ffp-contract=off")
  set_source_files_properties(src/kernels_avx512_vbmi.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vbmi;-mavx512vnni;-mavx2;-mfma;-ffp-contract=off")
endif()
if(MSVC)
  target_compile_options(boxwood PRIVATE /W4)
else()
  target_compile_options(boxwood PRIVATE -Wall -Wextra -Wpedantic)
  if(BOXWOOD_SANITIZE)
    target_compile_options(boxwood PUBLIC -fsanitize=address,undefined)
    target_link_options(boxwood PUBLIC -fsanitize=address,undefined)
  endif()
endif()

if(BOXWOOD_TESTS)
  enable_testing()
  add_executable(boxwood_test tests/cpp/boxwood_test.cpp)
  target_link_libraries(boxwood_test PRIVATE boxwood)
  target_include_directories(boxwood_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  if(NOT MSVC)
    target_compile_options(boxwood_test PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  add_test(NAME boxwood_test COMMAND boxwood_test)
  if(CMAKE_NM AND NOT MSVC)
    add_test(NAME simd_symbols COMMAND "${CMAKE_COMMAND}" -DNM=${CMAKE_NM}
             "-DOBJECTS=$<TARGET_OBJECTS:boxwood>" -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_simd_symbols.cmake")
  endif()
endif()

if(BOXWOOD_PYTHON)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE nanobind_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE nanobind_result)
  if(NOT nanobind_result EQUAL 0)
    message(FATAL_ERROR "${Python_EXECUTABLE} -m nanobind --cmake_dir failed; "
                       "install nanobind into that interpreter")
  endif()
  find_package(nanobind CONFIG REQUIRED HINTS "${nanobind_DIR}")

  nanobind_add_module(_boxwood NB_STATIC python/src/binding.cpp)
  target_link_libraries(_boxwood PRIVATE boxwood)

  # The package lives in the build tree only: build/<preset>/python/boxwood holds
  # the module and the copied .py files, so PYTHONPATH=build/<preset>/python works.
  set(BOXWOOD_PY_PKG "${CMAKE_BINARY_DIR}/python/boxwood")
  set_target_properties(_boxwood PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${BOXWOOD_PY_PKG}"
    RUNTIME_OUTPUT_DIRECTORY "${BOXWOOD_PY_PKG}")

  file(GLOB BOXWOOD_PY_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/python/boxwood/*.py")
  # add_custom_command(TARGET) takes no DEPENDS, so let the binding object depend
  # on the .py files: editing one recompiles, relinks and re-runs the copy.
  set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/python/src/binding.cpp"
    PROPERTIES OBJECT_DEPENDS "${BOXWOOD_PY_SOURCES}")
  add_custom_command(TARGET _boxwood POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            ${BOXWOOD_PY_SOURCES} "${BOXWOOD_PY_PKG}"
    COMMENT "Copying python/boxwood/*.py to ${BOXWOOD_PY_PKG}")

  # pip (pyproject.toml, scikit-build-core) puts the .py files into the wheel
  # itself and takes the module from this install rule.
  install(TARGETS _boxwood LIBRARY DESTINATION boxwood)
endif()

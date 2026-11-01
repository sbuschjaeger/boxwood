# Fails if an object file of a SIMD backend defines a weak function (nm type
# W): an inline or template function compiled with that file's SIMD flags,
# which the linker may pick for the whole program (see src/simd.hpp).
# Usage: cmake -DNM=<nm> -DOBJECTS=<objects, ;-separated> -P check_simd_symbols.cmake
set(found "")
foreach(obj IN LISTS OBJECTS)
  if(NOT obj MATCHES "kernels_avx")
    continue()
  endif()
  execute_process(COMMAND "${NM}" -C "${obj}" OUTPUT_VARIABLE symbols RESULT_VARIABLE result)
  if(NOT result EQUAL 0)
    message(FATAL_ERROR "${NM} failed on ${obj}")
  endif()
  string(REGEX MATCHALL "[^\n]* W [^\n]*" weak "${symbols}")
  foreach(line IN LISTS weak)
    string(APPEND found "${obj}: ${line}\n")
  endforeach()
endforeach()
if(found)
  message(FATAL_ERROR "weak symbols in SIMD object files:\n${found}")
endif()

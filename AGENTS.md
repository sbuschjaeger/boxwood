# Agent instructions (read before coding)

boxwood is a C++20 library for oblivious tree ensembles imported from CatBoost, with a nanobind Python package.
How it works is in `docs/site/explanations`.

- Python: use `.venv/bin/python` of this repository (`cmake --preset` configures with it; a symlink to another
  venv works).
- Build directories: `build/<preset>` only (`cmake --preset <name>`).
- Tests: `(cd build/<preset> && ctest)` and `.venv/bin/python -m pytest tests` (needs `build/release`).

## Rules for the C++ code

- C++20 and the standard library. No other dependencies except nanobind (Python binding only).
- The numeric rules of `docs/site/explanations/numerics.md` apply to every function that sums, fits or compares
  with CatBoost: every backend returns the same bits.
- Errors only through `BW_CHECK(condition, message)`, only at the API boundary. Kernels and hot loops do not check
  and do not throw.
- Must build with GCC, Clang and MSVC. No compiler extensions outside the SIMD backend files.
- No undefined behavior: no unaligned pointer casts (use `memcpy`), no signed overflow.

## Code style (all code, also tests)

- Keep the code small and easy to read. Write the simplest code that does the task. Make a helper or an
  abstraction only when two places use it. No dead code, no placeholder files for later work.
- One fact in one place: if two functions must agree, let them share the code.
- Comments explain how and why, for a reader who does not know the design documents. Do not cite design documents
  or milestones in code or comments. Do not write a comment that only repeats the code.
- Each public C++ function has a `///` doc comment: what it does, the arguments, the errors.
  Each public Python function and class has a Google-style docstring (`Args:`, `Returns:`, `Raises:`).

## When you finish

Print a short summary: files created and changed, how you tested (commands and results), open problems.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "boxwood/error.hpp"

namespace boxwood {

/// A JSON value: null, bool, number (double), string, array or object.
/// Object members keep their order from the input.
class Json {
 public:
  enum class Type : uint8_t { Null, Bool, Number, String, Array, Object };

  Type type = Type::Null;
  bool bool_value = false;
  double number = 0.0;
  std::string string;
  std::vector<Json> array;
  // Object members as two parallel vectors: a std::pair<std::string, Json>
  // member would need Json to be complete here, a std::vector<Json> does not.
  std::vector<std::string> object_keys;
  std::vector<Json> object_values;

  bool is_null() const { return type == Type::Null; }

  /// Number of elements (array), members (object) or bytes (string); 0 otherwise.
  size_t size() const;

  /// Member of an object by key. Throws Error if this is not an object or the key
  /// is missing.
  const Json& at(std::string_view key) const;

  /// Member of an object by key, or nullptr if this is not an object or the key
  /// is missing.
  const Json* find(std::string_view key) const;

  /// Element of an array. Throws Error if this is not an array or the index is
  /// out of range.
  const Json& operator[](size_t i) const;

  /// The number. Throws Error if this is not a number.
  double as_double() const;

  /// The string. Throws Error if this is not a string.
  const std::string& as_string() const;
};

/// Parse a JSON document. Throws Error with the byte offset on syntax errors.
Json parse_json(std::string_view text);

}  // namespace boxwood

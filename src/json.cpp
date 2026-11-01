#include "json.hpp"

#include <charconv>
#include <cstring>
#include <utility>

namespace boxwood {
namespace {

/// Recursive-descent parser over the whole document.
class Parser {
 public:
  explicit Parser(std::string_view text) : s_(text) {}

  Json parse() {
    skip_ws();
    Json v = value();
    skip_ws();
    BW_CHECK(i_ == s_.size(), msg("expected end of input"));
    return v;
  }

 private:
  std::string msg(std::string_view what) const {
    return "JSON syntax error at byte " + std::to_string(i_) + ": " + std::string(what);
  }
  [[noreturn]] void fail(std::string_view what) const { throw Error(msg(what)); }

  void skip_ws() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) ++i_;
  }

  bool eat(char c) {
    if (i_ < s_.size() && s_[i_] == c) {
      ++i_;
      return true;
    }
    return false;
  }

  void literal(const char* lit) {
    const size_t n = std::strlen(lit);
    BW_CHECK(s_.substr(i_, n) == lit, msg(std::string("expected '") + lit + "'"));
    i_ += n;
  }

  Json value() {
    BW_CHECK(i_ < s_.size(), msg("expected a value"));
    Json v;
    switch (s_[i_]) {
      case '{':
        v.type = Json::Type::Object;
        object_into(v);
        return v;
      case '[':
        v.type = Json::Type::Array;
        array_into(v);
        return v;
      case '"':
        v.type = Json::Type::String;
        v.string = string_raw();
        return v;
      case 't':
        literal("true");
        v.type = Json::Type::Bool;
        v.bool_value = true;
        return v;
      case 'f':
        literal("false");
        v.type = Json::Type::Bool;
        v.bool_value = false;
        return v;
      case 'n':
        literal("null");
        return v;
      default:
        v.type = Json::Type::Number;
        v.number = number();
        return v;
    }
  }

  void object_into(Json& v) {
    ++i_;  // '{'
    skip_ws();
    if (eat('}')) return;
    while (true) {
      BW_CHECK(i_ < s_.size() && s_[i_] == '"', msg("expected a string key"));
      std::string key = string_raw();
      skip_ws();
      BW_CHECK(eat(':'), msg("expected ':'"));
      skip_ws();
      v.object_keys.push_back(std::move(key));
      v.object_values.push_back(value());
      skip_ws();
      if (eat(',')) {
        skip_ws();
        continue;
      }
      BW_CHECK(eat('}'), msg("expected ',' or '}'"));
      return;
    }
  }

  void array_into(Json& v) {
    ++i_;  // '['
    skip_ws();
    if (eat(']')) return;
    while (true) {
      skip_ws();
      v.array.push_back(value());
      skip_ws();
      if (eat(',')) continue;
      BW_CHECK(eat(']'), msg("expected ',' or ']'"));
      return;
    }
  }

  /// Parses a string starting at the opening quote.
  std::string string_raw() {
    ++i_;  // opening '"'
    std::string out;
    while (true) {
      BW_CHECK(i_ < s_.size(), msg("unterminated string"));
      const char c = s_[i_++];
      if (c == '"') return out;
      if (c != '\\') {
        BW_CHECK(static_cast<unsigned char>(c) >= 0x20, msg("control character in string"));
        out += c;
        continue;
      }
      BW_CHECK(i_ < s_.size(), msg("unterminated escape"));
      const char e = s_[i_++];
      switch (e) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          uint32_t cp = hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF) {  // high surrogate: a low one must follow
            BW_CHECK(i_ + 1 < s_.size() && s_[i_] == '\\' && s_[i_ + 1] == 'u',
                     msg("missing low surrogate"));
            i_ += 2;
            const uint32_t lo = hex4();
            BW_CHECK(lo >= 0xDC00 && lo <= 0xDFFF, msg("invalid low surrogate"));
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          utf8(out, cp);
          break;
        }
        default: fail("invalid escape");
      }
    }
  }

  uint32_t hex4() {
    BW_CHECK(i_ + 4 <= s_.size(), msg("truncated \\u escape"));
    uint32_t v = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = s_[i_++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
      else {
        --i_;
        fail("invalid hex digit in \\u escape");
      }
    }
    return v;
  }

  void utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xC0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xE0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
      out += static_cast<char>(0xF0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
      out += static_cast<char>(0x80 | (cp & 0x3F));
    }
  }

  double number() {
    const char* first = s_.data() + i_;
    const char* last = s_.data() + s_.size();
    double v = 0.0;
    const std::from_chars_result res = std::from_chars(first, last, v);
    BW_CHECK(res.ec == std::errc{}, msg("invalid number"));
    i_ += static_cast<size_t>(res.ptr - first);
    return v;
  }

  std::string_view s_;
  size_t i_ = 0;
};

}  // namespace

size_t Json::size() const {
  switch (type) {
    case Type::Array: return array.size();
    case Type::Object: return object_keys.size();
    case Type::String: return string.size();
    default: return 0;
  }
}

const Json* Json::find(std::string_view key) const {
  if (type != Type::Object) return nullptr;
  for (size_t i = 0; i < object_keys.size(); ++i) {
    if (object_keys[i] == key) return &object_values[i];
  }
  return nullptr;
}

const Json& Json::at(std::string_view key) const {
  const Json* p = find(key);
  BW_CHECK(p != nullptr, "missing JSON key: " + std::string(key));
  return *p;
}

const Json& Json::operator[](size_t i) const {
  BW_CHECK(type == Type::Array, "JSON value is not an array");
  BW_CHECK(i < array.size(), "JSON array index out of range");
  return array[i];
}

double Json::as_double() const {
  BW_CHECK(type == Type::Number, "JSON value is not a number");
  return number;
}

const std::string& Json::as_string() const {
  BW_CHECK(type == Type::String, "JSON value is not a string");
  return string;
}

Json parse_json(std::string_view text) { return Parser(text).parse(); }

}  // namespace boxwood

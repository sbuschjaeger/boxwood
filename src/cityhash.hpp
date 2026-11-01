#pragma once

#include <cstdint>
#include <string_view>

namespace boxwood {

/// CityHash64 of the bytes of `s`, Yandex's variant (util/digest/city.cpp,
/// "CityHash Version 1"), the one CatBoost hashes category strings with. It
/// differs from Google's original CityHash for some string lengths.
uint64_t city_hash64(std::string_view s);

/// CatBoost's category hash: the low 32 bits of city_hash64 of the UTF-8 bytes.
uint32_t cat_hash(std::string_view s);

}  // namespace boxwood

#include "util/version.h"

namespace inferc {

// Anchor TU so version.h's constexpr has a place to live for ODR-use cases.
const std::string_view& VersionString() {
  return kVersion;
}

}  // namespace inferc

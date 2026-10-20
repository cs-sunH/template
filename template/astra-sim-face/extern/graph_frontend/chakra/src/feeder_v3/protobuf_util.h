#ifndef CHAKRA_FEEDER_V3_PROTOBUF_UTIL_H
#define CHAKRA_FEEDER_V3_PROTOBUF_UTIL_H

#include <cstdint>
#include <iostream>
#include <mutex>
#include "common.h"

namespace Chakra {
namespace FeederV3 {
class ProtobufUtils {
 public:
  static bool readVarint32(std::istream& f, uint32_t& value) {
    std::unique_lock<std::mutex> lock(_mutex);
    uint8_t byte;
    value = 0;
    int8_t shift = 0;
    while (f.read(reinterpret_cast<char*>(&byte), 1)) {
      value |= (byte & 0x7f) << shift;
      if (!(byte & 0x80))
        return true;
      shift += 7;
      if (shift > 28)
        return false;
    }
    return false;
  }

  template <typename T>
  static bool readMessage(std::istream& f, T& msg) {
    std::unique_lock<std::mutex> lock(_mutex);
    if (f.eof())
      return false;
    // Message-local buffer, not a static shared one: a short read must never
    // be able to expose another call's stale bytes to the parser.
    char buffer[DEFAULT_PROTOBUF_BUFFER_SIZE];
    uint32_t size;
    lock.unlock();
    if (!readVarint32(f, size))
      return false;
    lock.lock();
    char* buffer_use = buffer;
    if (size > DEFAULT_PROTOBUF_BUFFER_SIZE - 1) {
      // buffer is not large enough, use a dynamic buffer
      buffer_use = new char[size + 1];
    }
    f.read(buffer_use, size);
    const std::streamsize bytes_read = f.gcount();
    // Fail closed: a short read (truncated/corrupt stream) or a failed parse
    // must return false instead of silently handing back partial data.
    const bool size_complete =
        bytes_read == static_cast<std::streamsize>(size);
    const bool parse_ok =
        size_complete && msg.ParseFromArray(buffer_use, static_cast<int>(size));
    if (size > DEFAULT_PROTOBUF_BUFFER_SIZE - 1) {
      delete[] buffer_use;
    }
    return size_complete && parse_ok;
  }

 private:
  static std::mutex _mutex;
};

std::mutex ProtobufUtils::_mutex;
} // namespace FeederV3
} // namespace Chakra
#endif

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
      // unsigned arithmetic: a signed int shift at shift == 28 is UB
      value |= static_cast<uint32_t>(byte & 0x7f) << shift;
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
    static char buffer[DEFAULT_PROTOBUF_BUFFER_SIZE];
    uint32_t size;
    lock.unlock();
    if (!readVarint32(f, size))
      return false;
    lock.lock();
    if (size > MAX_PROTOBUF_MESSAGE_SIZE)
      return false;
    char* buffer_use = buffer;
    if (size > DEFAULT_PROTOBUF_BUFFER_SIZE - 1) {
      // buffer is not large enough, use a dynamic buffer.
      // size_t arithmetic: uint32 would wrap around at size == UINT32_MAX
      // and allocate 0 bytes, turning buffer_use[size] into a wild write.
      buffer_use = new char[static_cast<size_t>(size) + 1];
    }
    if (!f.read(buffer_use, size).good() ||
        f.gcount() != static_cast<std::streamsize>(size)) {
      // truncated message: never parse leftover bytes of a previous message
      if (buffer_use != buffer)
        delete[] buffer_use;
      return false;
    }
    buffer_use[size] = 0;
    if (!msg.ParseFromArray(buffer_use, size)) {
      // corrupted message: report instead of silently propagating defaults
      if (buffer_use != buffer)
        delete[] buffer_use;
      return false;
    }
    if (size > DEFAULT_PROTOBUF_BUFFER_SIZE - 1) {
      delete[] buffer_use;
    }
    return true;
  }

 private:
  // inline: an out-of-class definition in this header would be a redefinition
  // as soon as any second translation unit includes it (ODR)
  static inline std::mutex _mutex;
};

} // namespace FeederV3
} // namespace Chakra
#endif

#ifndef CHAKRA_FEEDER_V3_PROTOBUF_UTIL_H
#define CHAKRA_FEEDER_V3_PROTOBUF_UTIL_H

#include <cstdint>
#include <iostream>
#include <mutex>
#include <stdexcept>
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
    // Probe with peek() before consuming anything: peek() hitting EOF means
    // the previous message ended exactly at the end of the file -- a clean
    // boundary EOF. peek() sets eofbit in that case; the flag is left in
    // place on purpose so callers can test f.eof() and treat this false
    // return as a clean end of trace.
    if (f.peek() == std::istream::traits_type::eof())
      return false;
    static char buffer[DEFAULT_PROTOBUF_BUFFER_SIZE];
    uint32_t size;
    lock.unlock();
    if (!readVarint32(f, size)) {
      // at least one byte was available yet the varint size header could not
      // be read to completion: the file ends mid-header (or the varint is
      // malformed, > 5 bytes) -- a truncated tail, never a clean end
      throw std::runtime_error(
          "ProtobufUtils::readMessage: truncated or malformed varint size "
          "header, file might be truncated or corrupted");
    }
    lock.lock();
    if (size == 0) {
      // a bare 0x00 length prefix would parse as an all-default message
      // (id=0, type=INVALID) silently poisoning the node index
      throw std::runtime_error(
          "ProtobufUtils::readMessage: zero-length message near offset " +
          std::to_string(static_cast<std::streamoff>(f.tellg())) +
          ", file might be corrupted");
    }
    if (size > MAX_PROTOBUF_MESSAGE_SIZE)
      return false;
    char* buffer_use = buffer;
    if (size > DEFAULT_PROTOBUF_BUFFER_SIZE - 1) {
      // buffer is not large enough, use a dynamic buffer.
      // size_t arithmetic: uint32 would wrap around at size == UINT32_MAX
      // and allocate 0 bytes, turning buffer_use[size] into a wild write.
      buffer_use = new char[static_cast<size_t>(size) + 1];
    }
    f.read(buffer_use, size);
    if (f.gcount() != static_cast<std::streamsize>(size)) {
      // truncated message body: the file ends mid-message. Never parse
      // leftover bytes and never report a clean end -- hard error.
      if (buffer_use != buffer)
        delete[] buffer_use;
      throw std::runtime_error(
          "ProtobufUtils::readMessage: truncated message body, read " +
          std::to_string(f.gcount()) + " of " + std::to_string(size) +
          " bytes, file might be truncated or corrupted");
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

  static bool writeVarint32(std::ostream& f, uint32_t value) {
    std::unique_lock<std::mutex> lock(_mutex);
    uint8_t byte;
    while (value > 0x7f) {
      byte = (value & 0x7f) | 0x80;
      f.write(reinterpret_cast<char*>(&byte), 1);
      value >>= 7;
    }
    byte = value;
    f.write(reinterpret_cast<char*>(&byte), 1);
    return true;
  }

  template <typename T>
  static bool writeMessage(std::ostream& f, T& msg) {
    std::unique_lock<std::mutex> lock(_mutex);
    static char buffer[DEFAULT_PROTOBUF_BUFFER_SIZE];
    size_t size = msg.ByteSizeLong();
    if (size > DEFAULT_PROTOBUF_BUFFER_SIZE - 1) {
      // buffer is not large enough, use a dynamic buffer
      char* buffer_use = new char[size];
      msg.SerializeToArray(buffer_use, size);
      lock.unlock();
      writeVarint32(f, size);
      lock.lock();
      f.write(buffer_use, size);
      delete[] buffer_use;
    } else {
      msg.SerializeToArray(buffer, size);
      lock.unlock();
      writeVarint32(f, size);
      lock.lock();
      f.write(buffer, size);
    }
    return true;
  }

 private:
  static std::mutex _mutex;
};

std::mutex ProtobufUtils::_mutex;
} // namespace FeederV3
} // namespace Chakra
#endif

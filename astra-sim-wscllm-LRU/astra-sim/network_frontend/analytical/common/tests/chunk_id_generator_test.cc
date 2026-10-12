#include "common/ChunkIdGenerator.hh"

#include <cstdio>
#include <cstdlib>

using namespace AstraSimAnalytical;

namespace {

void expect(const bool condition, const char* const message) {
    if (!condition) {
        std::fprintf(stderr, "[chunk-id-generator] FAIL: %s\n", message);
        std::abort();
    }
}

}  // namespace

int main() {
    ChunkIdGenerator generator;
    constexpr int tag = 17;
    constexpr int src = 2;
    constexpr int dest = 5;
    constexpr ChunkSize bytes = 4096;

    // Two sends under one legacy-reused key stay resident until both matched
    // callback pairs complete, even when their arrivals complete out of order.
    expect(generator.create_send_chunk_id(tag, src, dest, bytes) == 0,
           "first send id");
    expect(generator.create_send_chunk_id(tag, src, dest, bytes) == 1,
           "second send id");
    expect(generator.size() == 1, "same-key sends share one entry");
    expect(generator.create_recv_chunk_id(tag, src, dest, bytes) == 0,
           "first receive id");
    expect(generator.create_recv_chunk_id(tag, src, dest, bytes) == 1,
           "second receive id");
    generator.complete(tag, src, dest, bytes, 1);
    expect(generator.size() == 1, "entry retained after one completion");
    generator.complete(tag, src, dest, bytes, 0);
    expect(generator.size() == 0, "entry reclaimed after all completions");

    // recv-before-send uses the same id stream and also reclaims the key.
    expect(generator.create_recv_chunk_id(tag, src, dest, bytes) == 0,
           "recv-first id");
    expect(generator.size() == 1, "recv-first creates an entry");
    expect(generator.create_send_chunk_id(tag, src, dest, bytes) == 0,
           "send after recv-first id");
    generator.complete(tag, src, dest, bytes, 0);
    expect(generator.size() == 0, "recv-first entry reclaimed");

    std::printf("[chunk-id-generator] PASS\n");
    return 0;
}

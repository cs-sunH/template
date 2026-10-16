#pragma once
// Offline instrumentation only. Never included by production sources/builds.
#include <array>
#include <cstdint>
#include <time.h>
namespace update_trace {
enum Key { Update, Advance, Integrate, Settle, Allocate, Collect, Symmetry, Rows, Solver, Count };
inline constexpr std::array<const char*,Count> names{{"update","advance","integrate","settle",
    "allocate","collect","symmetry","rows","solver"}};
inline thread_local bool enabled=false;
inline thread_local std::array<std::uint64_t,Count> totals{},calls{};
inline std::uint64_t now(){timespec t{};clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t);return t.tv_sec*1000000000ULL+t.tv_nsec;}
struct Scope {
  Key key;bool active;std::uint64_t start;
  explicit Scope(Key k):key(k),active(enabled),start(active?now():0){}
  ~Scope(){if(active){totals[key]+=now()-start;++calls[key];}}
};
struct Timeline {
  bool active;std::uint64_t start;
  Timeline():active(enabled),start(active?now():0){}
  void end(Key key){if(active){const auto t=now();totals[key]+=t-start;++calls[key];start=t;}}
};
}

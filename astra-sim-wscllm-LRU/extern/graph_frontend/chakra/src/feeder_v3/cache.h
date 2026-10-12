#ifndef CHAKRA_FEEDER_V3_CACHE_H
#define CHAKRA_FEEDER_V3_CACHE_H

#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>

namespace Chakra {
namespace FeederV3 {

template <typename K, typename V>
class Cache {
 public:
  Cache(size_t capacity) : capacity(capacity) {
    // put()'s eviction loop would dereference an empty lru forever when
    // capacity == 0; reject that degenerate configuration up front.
    if (capacity == 0) {
      throw std::runtime_error("Cache capacity must be positive");
    }
  }
  void put(const K& key, const V& value) {
    std::unique_lock lock(cache_mutex);
    if (this->cache.find(key) != this->cache.end()) {
      // hit and update
      this->lru.erase(this->cache[key].second);
      this->lru.push_back(key);
      this->cache[key].second = --this->lru.end();
      this->cache[key].first = std::make_shared<V>(value);
    } else {
      // miss
      while (this->cache.size() >= this->capacity) {
        // evict
        auto victim = this->lru.front();
        this->cache.erase(victim);
        this->lru.pop_front();
      }
      // and put new
      this->lru.push_back(key);
      this->cache[key] =
          std::make_pair(std::make_shared<V>(value), --this->lru.end());
    }
  }
  std::shared_ptr<const V> get_locked(const K& key) {
    std::shared_lock lock(cache_mutex);
    if (this->cache.find(key) == this->cache.end()) {
      throw std::runtime_error("Key not found in cache");
    }
    return this->cache.at(key).first;
  }
  std::shared_ptr<const V> get_or_null_locked(const K& key) {
    std::shared_lock lock(cache_mutex);
    if (this->cache.find(key) == this->cache.end()) {
      return std::shared_ptr<const V>();
    }
    return this->cache.at(key).first;
  }

  ~Cache() {
    this->cache.clear();
    this->lru.clear();
  }

 private:
  size_t capacity;
  std::unordered_map<
      K,
      std::pair<std::shared_ptr<V>, typename std::list<K>::iterator>>
      cache;
  std::list<K> lru;
  mutable std::shared_mutex cache_mutex;
};

} // namespace FeederV3
} // namespace Chakra

#endif

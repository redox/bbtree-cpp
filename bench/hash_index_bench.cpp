// `shoal::unsync::HashIndex<uint64_t>` as a join or aggregate build uses it:
// prehashed rows, 2048 a call, into an index reserved to its final size.
// Every row of the insert cell is a new key: one reaches `compare` only past
// another key of its fingerprint, and no growth calls `stored_hash`. The hit
// cell then finds the same keys, so that the two cells' difference is what a
// new key costs beyond a probe. The keys are random words, used as their own
// hashes (design/05-hash-maps.md, "Building on new keys").
//
// The pointer cell inserts the same rows into a table after DuckDB's join
// and aggregate pointer table, at the index's capacity: a slot is one word,
// a 16-bit salt over a 48-bit row pointer, probed linearly, so a new key
// reads and writes one random line, where the index's touches its tag's and
// its descriptor's.
//
// usage: hash_index_bench [entries] [rows a cell]
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "shoal/unsync/hash_index.hpp"
#include "workload.hpp"

using shoal_bench::Keys;
using shoal_bench::Stopwatch;
using Index = shoal::unsync::HashIndex<std::uint64_t>;

// The reference: DuckDB's salt and pointer slots, inserting rows that are
// all new keys, a row at a time as its build loop does.
class PointerTable {
 public:
  explicit PointerTable(std::size_t capacity) : slots_(capacity), mask_(capacity - 1) {}
  void clear() { std::fill(slots_.begin(), slots_.end(), 0); }
  void insert_new(std::uint64_t hash, std::uint64_t row) {
    const std::uint64_t salt = hash & ~pointer_mask;
    for (std::size_t slot = hash & mask_;; slot = (slot + 1) & mask_) {
      const std::uint64_t entry = slots_[slot];
      if (entry == 0) {
        slots_[slot] = salt | row;
        return;
      }
      // A salt match would compare rows; new keys never match, and probe on.
    }
  }

 private:
  static constexpr std::uint64_t pointer_mask = (std::uint64_t{1} << 48) - 1;
  std::vector<std::uint64_t> slots_;
  std::size_t mask_;
};

int main(int argc, char** argv) {
  const std::size_t entries = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 2000000;
  constexpr std::size_t per_call = 2048;
  const std::size_t row_count = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 10000000;
  // Small indexes fill again and again, so each cell times as many rows as
  // a large one does.
  const std::size_t fills = entries < row_count ? row_count / entries : 1;
  const std::vector<std::uint64_t> hashes = Keys::present(entries);
  std::vector<std::uint64_t> results(per_call), positions(per_call);
  // A new key reaches compare only past another key of its fingerprint,
  // which it never matches, and, reserved, never reaches stored_hash.
  std::size_t candidates = 0;
  const auto reject_all = [&](std::uint32_t* rows, std::size_t count, std::uint64_t*,
                              std::uint32_t* rejected) {
    candidates += count;
    std::copy(rows, rows + count, rejected);
    return count;
  };
  const auto unreached_stored_hash = [](std::uint64_t) -> std::uint64_t { std::abort(); };
  Index index;
  Index::ProbeState scratch;
  index.reserve(entries, unreached_stored_hash);
  double insert_ns = 0;
  for (std::size_t fill = 0; fill < fills; ++fill) {
    index.clear_retain_capacity();
    Stopwatch watch;
    for (std::size_t base = 0; base < entries; base += per_call) {
      const std::size_t width = entries - base < per_call ? entries - base : per_call;
      index.find_or_create_many(
          scratch, width, [&](std::uint32_t row) { return hashes[base + row]; },
          unreached_stored_hash, reject_all,
          [&](const std::uint32_t* rows, std::size_t count, std::uint64_t* handles) {
            for (std::size_t i = 0; i < count; ++i) handles[i] = base + rows[i] + 1;
          },
          results.data(), positions.data());
      shoal_bench::keep(results[0]);
    }
    insert_ns += watch.ns_per(1);
  }
  if (index.size() != entries) std::abort();
  shoal_bench::report("u64 index_insert_new", entries, insert_ns / static_cast<double>(fills * entries));
  std::printf("  candidates a row %.6f\n", static_cast<double>(candidates) / static_cast<double>(fills * entries));
  {
    Stopwatch watch;
    std::uint64_t sum = 0;
    for (std::size_t fill = 0; fill < fills; ++fill)
      for (std::size_t base = 0; base < entries; base += per_call) {
        const std::size_t width = entries - base < per_call ? entries - base : per_call;
        index.find_many(
            scratch, width, [&](std::uint32_t row) { return hashes[base + row]; },
            [&](std::uint32_t* candidates, std::size_t count, std::uint64_t* found,
                std::uint32_t* rejected) {
              std::size_t n = 0;
              for (std::size_t i = 0; i < count; ++i)
                if (found[candidates[i]] != base + candidates[i] + 1) rejected[n++] = candidates[i];
              return n;
            },
            results.data());
        sum += results[0];
      }
    shoal_bench::keep(sum);
    shoal_bench::report("u64 index_find_hit", entries, watch.ns_per(fills * entries));
  }
  {
    PointerTable table(index.capacity());
    double pointer_ns = 0;
    for (std::size_t fill = 0; fill < fills; ++fill) {
      table.clear();
      Stopwatch watch;
      for (std::size_t row = 0; row < entries; ++row) table.insert_new(hashes[row], row + 1);
      pointer_ns += watch.ns_per(1);
    }
    shoal_bench::report("u64 pointer_insert_new", entries,
                        pointer_ns / static_cast<double>(fills * entries));
  }
  return 0;
}

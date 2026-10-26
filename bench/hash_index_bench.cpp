// `shoal::unsync::HashIndex` as a join or aggregate build uses it: prehashed
// rows, 2048 a call, into an index reserved to its final size, every row a
// new key (design/05-hash-maps.md, "Building on new keys"). A new key
// reaches `compare` only past another key of its fingerprint, and no growth
// calls `stored_hash`. The keys are random words, used as their own hashes.
//
// - index_insert_new: `find_or_create_many`, `create` returning handles, as
//   an aggregate build materializes rows only for the keys it misses.
// - index_insert_known: `find_or_insert_many` (D34), hashes and handles from
//   contiguous arrays, no positions.
// - index_insert_rows: the same, shaped as DuckDB's join build calls it:
//   handles point at 32-byte external rows, positions are requested (a
//   duplicate needs its position for `replace_at`), and after each call the
//   build writes every row's chain pointer.
// - index_create_mixed, index_insert_mixed: `find_or_create_many` and
//   `find_or_insert_many` on rows whose keys are drawn from half as many, so
//   that about 43% of rows are new keys, in random order: where whether a
//   row is new can't be predicted.
// - index_find_hit: `find_many` on the same keys.
// - pointer_insert_new, pointer_insert_rows: the same rows into a table
//   after DuckDB's join pointer table, at the index's capacity, a row at a
//   time as its build loop inserts: a slot is one word, a 16-bit salt over a
//   48-bit row pointer, probed linearly. The rows cell also writes each
//   row's chain pointer, the slot's previous word, as DuckDB's build does.
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

// A build row stored outside the index, as a join's row layout holds one:
// its key, its hash, the chain pointer to the previous row of its key, and
// a payload.
struct Row {
  std::uint64_t key;
  std::uint64_t hash;
  Row* next;
  std::uint64_t payload;
};

// The reference: DuckDB's salt and pointer slots, inserting rows that are
// all new keys, a row at a time as its build loop does.
class PointerTable {
 public:
  explicit PointerTable(std::size_t capacity) : slots_(capacity), mask_(capacity - 1) {}
  void clear() { std::fill(slots_.begin(), slots_.end(), 0); }
  // Inserts a new key's row and returns the slot's previous word, the chain
  // pointer a duplicate would link to: zero, as no key repeats.
  std::uint64_t insert_new(std::uint64_t hash, std::uint64_t row) {
    const std::uint64_t salt = hash & ~pointer_mask;
    for (std::size_t slot = hash & mask_;; slot = (slot + 1) & mask_) {
      const std::uint64_t entry = slots_[slot];
      if (entry == 0) {
        slots_[slot] = salt | row;
        return entry;
      }
      // A salt match would compare rows; new keys never match, and probe on.
    }
  }

 private:
  static constexpr std::uint64_t pointer_mask = (std::uint64_t{1} << 48) - 1;
  std::vector<std::uint64_t> slots_;
  std::size_t mask_;
};

// Every cell on the same keys, each timing `fills` builds of `entries` rows.
class BuildCells {
 public:
  static constexpr std::size_t per_call = 2048;

 private:
  template <class Handle>
  static std::uint64_t unreached_stored_hash(Handle) {
    std::abort();
  }
  std::size_t timed_rows() const { return fills_ * entries_; }
  auto hash_at(std::size_t base) const {
    return [this, base](std::uint32_t row) { return hashes_[base + row]; };
  }
  auto mixed_hash_at(std::size_t base) const {
    return [this, base](std::uint32_t row) { return hashes_[mixed_keys_[base + row]]; };
  }
  // Accepts a candidate row (handle - 1) of the row's mixed key.
  auto same_mixed_key(std::size_t base) const {
    return [this, base](std::uint32_t* candidates, std::size_t count, std::uint64_t* found,
                        std::uint32_t* rejected) {
      std::size_t n = 0;
      for (std::size_t i = 0; i < count; ++i)
        if (mixed_keys_[found[candidates[i]] - 1] != mixed_keys_[base + candidates[i]])
          rejected[n++] = candidates[i];
      return n;
    };
  }
  // A new key reaches compare only past another key of its fingerprint,
  // which it never matches.
  template <class Handle>
  auto reject_all(std::size_t width) {
    new_key_rows_ += width;
    return [this](std::uint32_t* rows, std::size_t count, Handle*, std::uint32_t* rejected) {
      candidates_ += count;
      std::copy(rows, rows + count, rejected);
      return count;
    };
  }

 public:

  BuildCells(std::size_t entries, std::size_t row_count)
      : entries_(entries),
        // Small indexes fill again and again, so each cell times as many
        // rows as a large one does.
        fills_(entries < row_count ? row_count / entries : 1),
        hashes_(Keys::present(entries)),
        rows_(entries),
        row_handles_(entries),
        handles_(entries),
        mixed_keys_(entries) {
    for (std::size_t row = 0; row < entries; ++row) {
      rows_[row] = {row, hashes_[row], nullptr, row};
      row_handles_[row] = &rows_[row];
      handles_[row] = row + 1;
      mixed_keys_[row] = Keys::mix(row ^ 0x5555555555555555ull) % (entries / 2 + 1);
    }
    numbered_.reserve(entries, unreached_stored_hash<std::uint64_t>);
    pointed_.reserve(entries, unreached_stored_hash<Row*>);
  }

  std::size_t entries() const { return entries_; }
  std::size_t capacity() const { return numbered_.capacity(); }
  double candidates_a_row() const {
    return static_cast<double>(candidates_) / static_cast<double>(new_key_rows_);
  }

  double insert_new() {
    return time_builds([&] { numbered_.clear_retain_capacity(); }, [&](std::size_t base, std::size_t width) {
      numbered_.find_or_create_many(
          scratch_, width, hash_at(base), unreached_stored_hash<std::uint64_t>,
          reject_all<std::uint64_t>(width),
          [&](const std::uint32_t* rows, std::size_t count, std::uint64_t* out) {
            for (std::size_t i = 0; i < count; ++i) out[i] = base + rows[i] + 1;
          },
          results_.data(), positions_.data());
      shoal_bench::keep(results_[0]);
    });
  }

  double insert_known() {
    return time_builds([&] { numbered_.clear_retain_capacity(); }, [&](std::size_t base, std::size_t width) {
      numbered_.find_or_insert_many(scratch_, width, hash_at(base), unreached_stored_hash<std::uint64_t>,
                                    reject_all<std::uint64_t>(width), handles_.data() + base, results_.data());
      shoal_bench::keep(results_[0]);
    });
  }

  double insert_rows() {
    return time_builds([&] { pointed_.clear_retain_capacity(); }, [&](std::size_t base, std::size_t width) {
      pointed_.find_or_insert_many(pointed_scratch_, width, hash_at(base), unreached_stored_hash<Row*>,
                                   reject_all<Row*>(width), row_handles_.data() + base,
                                   row_results_.data(), positions_.data());
      for (std::size_t i = 0; i < width; ++i) {
        Row* const row = row_handles_[base + i];
        row->next = row_results_[i] == row ? nullptr : row_results_[i];
      }
    });
  }

  double create_mixed() {
    return time_builds([&] { numbered_.clear_retain_capacity(); }, [&](std::size_t base, std::size_t width) {
      numbered_.find_or_create_many(
          scratch_, width, mixed_hash_at(base), unreached_stored_hash<std::uint64_t>, same_mixed_key(base),
          [&](const std::uint32_t* rows, std::size_t count, std::uint64_t* out) {
            for (std::size_t i = 0; i < count; ++i) out[i] = base + rows[i] + 1;
          },
          results_.data(), positions_.data());
      shoal_bench::keep(results_[0]);
    });
  }

  double insert_mixed() {
    return time_builds([&] { numbered_.clear_retain_capacity(); }, [&](std::size_t base, std::size_t width) {
      numbered_.find_or_insert_many(scratch_, width, mixed_hash_at(base), unreached_stored_hash<std::uint64_t>,
                                    same_mixed_key(base), handles_.data() + base, results_.data(),
                                    positions_.data());
      shoal_bench::keep(results_[0]);
    });
  }

  double find_hit() {
    Stopwatch watch;
    std::uint64_t sum = 0;
    for (std::size_t fill = 0; fill < fills_; ++fill)
      for (std::size_t base = 0; base < entries_; base += per_call) {
        const std::size_t width = std::min(per_call, entries_ - base);
        numbered_.find_many(
            scratch_, width, hash_at(base),
            [&](std::uint32_t* candidates, std::size_t count, std::uint64_t* found,
                std::uint32_t* rejected) {
              std::size_t n = 0;
              for (std::size_t i = 0; i < count; ++i)
                if (found[candidates[i]] != base + candidates[i] + 1) rejected[n++] = candidates[i];
              return n;
            },
            results_.data());
        sum += results_[0];
      }
    shoal_bench::keep(sum);
    return watch.ns_per(timed_rows());
  }

  double pointer_new() {
    PointerTable table(capacity());
    return time_builds([&] { table.clear(); }, [&](std::size_t base, std::size_t width) {
      for (std::size_t row = base; row < base + width; ++row) table.insert_new(hashes_[row], row + 1);
    });
  }

  double pointer_rows() {
    PointerTable table(capacity());
    return time_builds([&] { table.clear(); }, [&](std::size_t base, std::size_t width) {
      for (std::size_t row = base; row < base + width; ++row) {
        Row* const handle = row_handles_[row];
        handle->next = reinterpret_cast<Row*>(
            table.insert_new(hashes_[row], reinterpret_cast<std::uintptr_t>(handle)));
      }
    });
  }

 private:
  // Times `fills_` builds, each emptied by `reset` untimed, then called a
  // batch at a time as call(base, width).
  template <class Reset, class Call>
  double time_builds(Reset reset, Call call) {
    double ns = 0;
    for (std::size_t fill = 0; fill < fills_; ++fill) {
      reset();
      Stopwatch watch;
      for (std::size_t base = 0; base < entries_; base += per_call) call(base, std::min(per_call, entries_ - base));
      ns += watch.ns_per(1);
    }
    return ns / static_cast<double>(timed_rows());
  }

  std::size_t entries_, fills_;
  std::vector<std::uint64_t> hashes_;
  std::vector<Row> rows_;
  std::vector<Row*> row_handles_;
  std::vector<std::uint64_t> handles_;
  std::vector<std::uint64_t> mixed_keys_;
  shoal::unsync::HashIndex<std::uint64_t> numbered_;
  shoal::unsync::HashIndex<Row*> pointed_;
  shoal::unsync::HashIndex<std::uint64_t>::ProbeState scratch_;
  shoal::unsync::HashIndex<Row*>::ProbeState pointed_scratch_;
  std::vector<std::uint64_t> results_ = std::vector<std::uint64_t>(per_call);
  std::vector<Row*> row_results_ = std::vector<Row*>(per_call);
  std::vector<std::uint64_t> positions_ = std::vector<std::uint64_t>(per_call);
  std::size_t candidates_ = 0;
  std::size_t new_key_rows_ = 0;  // rows of the cells that reject all
};

int main(int argc, char** argv) {
  const std::size_t entries = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 2000000;
  const std::size_t row_count = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 10000000;
  BuildCells cells(entries, row_count);
  shoal_bench::report("u64 index_insert_new", entries, cells.insert_new());
  shoal_bench::report("u64 index_insert_known", entries, cells.insert_known());
  shoal_bench::report("u64 index_insert_rows", entries, cells.insert_rows());
  shoal_bench::report("u64 index_create_mixed", entries, cells.create_mixed());
  shoal_bench::report("u64 index_insert_mixed", entries, cells.insert_mixed());
  // The index holds the last build's entries: rebuild the known ones to find.
  cells.insert_known();
  shoal_bench::report("u64 index_find_hit", entries, cells.find_hit());
  shoal_bench::report("u64 pointer_insert_new", entries, cells.pointer_new());
  shoal_bench::report("u64 pointer_insert_rows", entries, cells.pointer_rows());
  std::printf("  capacity %zu, candidates a row %.6f\n", cells.capacity(), cells.candidates_a_row());
  return 0;
}
